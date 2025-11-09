### [GPU 外部:GPU 如何與世界通訊](https://huggingfacetb-smol-training-playbook.hf.space/#outside-a-gpu-how-gpus-talk-to-the-world)

現在我們理解了 GPU 如何使用其內部記憶體層次結構執行計算,我們需要解決一個關鍵現實:GPU 不是孤立運作的。在任何計算發生之前,資料必須載入到 GPU 的記憶體中。CPU 需要排程 kernels 並協調工作。在分散式訓練中,GPU 必須不斷地與彼此交換啟動、梯度和模型權重。

![Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/h100_dgx_2891384e-bcac-80cf-9f86-ccf0653a79e5.B0aXfJMx_Z2pQ1Hl.webp)

DGX H100。來源:NVIDIA

這就是外部通訊基礎設施變得至關重要的地方。無論你的 GPU 計算單元有多強大,如果資料無法足夠快地到達它們,無論是來自 CPU、來自儲存還是來自其他 GPU,你昂貴的硬體就會閒置。理解這些通訊路徑及其頻寬特性對於最大化硬體利用率和最小化瓶頸至關重要。

在本節中,我們將查看連接 GPU 到外部世界的四個關鍵通訊連結:
  - **GPU-CPU**:CPU 如何排程工作並將資料傳輸到 GPU
  - **GPU-GPU intra-node**:同一機器上的 GPU 如何通訊
  - **GPU-GPU inter-node**:不同機器上的 GPU 如何透過網路通訊
  - **GPU-Storage**:資料如何從儲存流向 GPU 記憶體

這些連結中的每一個都有不同的頻寬和延遲特性,理解它們將幫助你識別訓練管線可能在哪裡出現瓶頸。為了讓這更容易理解,我們創建了一個簡化的圖表,突顯最重要的元件和通訊連結:

NodeCPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitchCPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitchNUMA 0NUMA 1

Show Real Bandwidths

Select path... Intranode: CPU ⟷ GPU Intranode: GPU ⟷ GPU via CPU Intranode: GPU ⟷ GPU via NVSwitch Intranode: GPU ⟷ GPU via EFA Internode: GPU ⟷ GPU via EFA Storage: GPU ⟷ Storage Storage: CPU ⟷ Storage Storage: GPU ⟷ Storage via CPU

EFA Link - 12.5 GB/sPCIe Gen4 - 16 GB/sPCIe Gen5 - 64 GB/sNVLink 4.0 - 900 GB/s

Bandwidth Max
for CPU → GPU
-
GB/s
-
Efficiency

Real Bandwidths

我們的 AWS p5 instance 設定中關鍵元件和通訊連結的簡化圖表

如果這看起來令人不知所措,不用擔心。我們將詳細深入每一個連接,並測量它們的實際頻寬以理解每個連結的效能特性。

#### [GPU-to-CPU](https://huggingfacetb-smol-training-playbook.hf.space/#gpu-to-cpu)

**TL;DR:** CPU 透過 PCIe 連接協調 GPU 工作,在我們的 p5 instance 中,CPU-to-GPU 傳輸的瓶頸在~14.2 GB/s(PCIe Gen4 x8)。CPU-GPU 延遲約為~1.4 微秒,這增加了對於有許多小 kernels 的工作負載來說有問題的 kernel 啟動開銷。CUDA Graphs 可以透過批次處理運算來減少這種開銷。NUMA 親和性在多插座系統上至關重要;在錯誤的 CPU 插座上執行 GPU 程序會增加顯著的延遲。現代架構如 Grace Hopper 透過 NVLink-C2C(900 GB/s vs 128 GB/s)消除了 PCIe 瓶頸。

CPU 是 GPU 計算的協調者。它負責啟動 kernels、管理記憶體分配和協調資料傳輸。但 CPU 實際上能以多快的速度與 GPU 通訊?這由它們之間的 **PCIe (Peripheral Component Interconnect Express)** 連接決定。

理解這個連結至關重要,因為它影響:
  - **Kernel 啟動延遲**:CPU 能多快在 GPU 上排程工作
  - **資料傳輸速度**:我們能多快在 CPU 和 GPU 記憶體之間移動資料
  - **同步開銷**:CPU-GPU 協調點的成本

在現代 GPU 伺服器中,CPU-GPU 連接已經顯著演變。雖然早期系統使用直接 PCIe 連接,但現代高效能系統如 DGX H100 使用更複雜的拓撲,使用 PCIe _switches_ 來有效管理多個 GPU。而在最新的 [GB200 架構](https://newsletter.semianalysis.com/p/nvidias-blackwell-reworked-shipment)中,NVIDIA 透過將 CPU 和 GPU 放在同一個印刷電路板上,完全消除了對外部 switches 的需求,進一步推進了這一點。

讓我們使用 `lstopo` 檢查我們的 p5 instance 的物理拓撲,然後測量這個關鍵連結的實際效能,以識別潛在的瓶頸。

```

$ lstopo -v
...
 HostBridge L#1 (buses=0000:[44-54])
    PCIBridge L#2 (busid=0000:44:00.0 id=1d0f:0200 class=0604(PCIBridge) link=15.75GB/s buses=0000:[45-54] PCISlot=64)
        PCIBridge L#3 (busid=0000:45:00.0 id=1d0f:0200 class=0604(PCIBridge) link=15.75GB/s buses=0000:[46-54] PCISlot=1-1)
            ...
            PCIBridge L#12 (busid=0000:46:01.4 id=1d0f:0200 class=0604(PCIBridge) link=63.02GB/s buses=0000:[53-53])
                PCI L#11 (busid=0000:53:00.0 id=10de:2330 class=0302(3D) link=63.02GB/s PCISlot=86-1)
                    Co-Processor(CUDA) L#8 (Backend=CUDA GPUVendor="NVIDIA Corporation" GPUModel="NVIDIA H100 80GB HBM3" CUDAGlobalMemorySize=83295872 CUDAL2CacheSize=51200 CUDAMultiProcessors=132 CUDACoresPerMP=128 CUDASharedMemorySizePerMP=48) "cuda0"
                    GPU(NVML) L#9 (Backend=NVML GPUVendor="NVIDIA Corporation" GPUModel="NVIDIA H100 80GB HBM3" NVIDIASerial=1654922006536 NVIDIAUUID=GPU-ba136838-6443-7991-9143-1bf4e48b2994) "nvml0"
            ...
...

```

從 `lstopo` 輸出中,我們可以看到系統中兩個關鍵的 PCIe 頻寬值:
  - **15.75GB/s**:對應於 PCIe Gen4 x8 連結(CPU 到 PCIe switches)
  - **63.02GB/s**:對應於 PCIe Gen5 x16 連結(PCIe switches 到 GPUs)

為了更好地理解整個拓撲,我們可以使用以下方式視覺化它:

```

$ lstopo --whole-system lstopo-diagram.png

```

![Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/lstopo_29c1384e-bcac-80c9-9715-cbfe9e73d86b.DCCvpO8Q_ZPxGsS.webp)

這個圖表展示了我們系統的階層結構:
  - 它包含兩個 **NUMA**(Non-Uniform Memory Access)節點(NUMA 是每個 CPU 插座的記憶體區域)
  - 每個 **CPU socket** 透過 **PCIe Gen4** x8 連結(15.75GB/s)連接到四個 **PCIe switches**
  - 每個 **PCIe switch** 透過 **PCIe Gen5** x16 連結(63.02GB/s)連接到一個 **H100 GPU**
  - …(我們將在下一節中探索其他元件,如 NVSwitch、EFA 網路卡和 NVMe 硬碟。)

PCIe 規格在世代之間不同,每代將每條通道的傳輸速率加倍。請注意,傳輸速率以 GT/s(每秒千兆傳輸)測量,代表原始訊號速率,而輸送量以 GB/s(每秒千兆位元組)測量,考慮編碼開銷並代表實際可用頻寬:

PCIe Version | Transfer Rate (per lane) | Throughput (GB/s)
---|---|---
×1 | ×2 | ×4
1.0 | 2.5 GT/s | 0.25
2.0 | 5.0 GT/s | 0.5
3.0 | 8.0 GT/s | 0.985
4.0 | 16.0 GT/s | 1.969
5.0 | 32.0 GT/s | 3.938
6.0 | 64.0 GT/s | 7.563
7.0 | 128.0 GT/s | 15.125

理論 PCIe 頻寬。來源:https://en.wikipedia.org/wiki/PCI_Express

CPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitch

EFA Link - 12.5 GB/sPCIe Gen4 - 16 GB/sPCIe Gen5 - 64 GB/sNVLink 4.0 - 900 GB/s

Bandwidth Max
for CPU ⟷ GPU
16
GB/s
-
Efficiency

Real Bandwidths

CPU-to-GPU 通訊路徑。

從拓撲圖和 PCIe 頻寬表中,我們可以看到 CPU-to-GPU 路徑經過兩個 PCIe 跳躍:首先從 CPU 到 PCIe switch 透過 **PCIe Gen4** x8(15.754 GB/s),然後從 PCIe switch 到 GPU 透過 **PCIe Gen5** x16(63.015 GB/s)。_這意味著 CPU-GPU 通訊的瓶頸是第一跳,為 15.754 GB/s_。讓我們用另一個工具 `nvbandwidth` 來驗證這一點!

`host_to_device_memcpy_ce` 命令使用 GPU 的複製引擎測量從 host (CPU) 記憶體到 device (GPU) 記憶體的 `cuMemcpyAsync` 頻寬。

```

./nvbandwidth -t host_to_device_memcpy_ce -b <message_size> -i 5

```

CPU-> GPU measured bandwidth
Reset View
Legend

使用 nvbandwidth 的 host_to_device 測試測量的 CPU-to-GPU 頻寬,顯示大型傳輸時 PCIe Gen4 x8 瓶頸約為~14.2 GB/s

結果確實顯示,對於小訊息大小,我們受延遲限制,但對於大訊息大小,我們實現了 **~14.2 GB/s**,這約為 PCIe Gen4 x8 理論 15.754 GB/s 頻寬的 90%。這證實了在 **CPU-GPU** 通訊中,CPU-to-PCIe switch 連結確實是我們的瓶頸。

除了頻寬,**延遲**對於 CPU-GPU 通訊同樣重要,因為它決定了我們能多快排程 kernels。為了測量這個,我們使用 `nvbandwidth` 的 `host_device_latency_sm` 測試,它使用 pointer-chase kernel 來測量往返延遲。`host_device_latency_sm` 測試透過在 host (CPU) 上分配緩衝區並從 GPU 使用 pointer-chase kernel 存取它來測量往返延遲。這模擬了 CPU-GPU 通訊的實際延遲。

```

./nvbandwidth -t host_device_latency_sm -i 5

```

CPU-> GPU measured latency
Reset View
Legend

使用 nvbandwidth 的 host_device_latency_sm 測試測量的 CPU-to-GPU 延遲(調整為使緩衝區大小可變),顯示約 1.4 微秒的往返延遲

結果顯示 **延遲**約為 **1.4 微秒**。這解釋了我們在 ML 工作負載中經常觀察到的幾微秒的 kernel 啟動開銷。對於啟動許多小 kernels 的工作負載,增加的延遲可能成為瓶頸;否則,開銷會被重疊執行隱藏。

例如,對於小型模型或小批次,我們可以看到推理在 GPU 上飽和,因為 kernel 啟動。FlashFormer 透過融合整個層來獲得加速來解決這個問題([Nrusimha et al., 2025](https://arxiv.org/abs/2505.22758))。

CUDA Graphs 用於減少啟動開銷

CUDA Graphs 可以透過捕獲一系列運算並將它們作為單個單元重放,顯著減少 kernel 啟動開銷,消除每次 kernel 啟動的 CPU-GPU 往返延遲的微秒數。這對於有許多小 kernels 或頻繁 CPU-GPU 同步的工作負載特別有益。有關理解和優化啟動開銷的更多詳細資訊,請參閱 [Understanding the Visualization of Overhead and Latency in NVIDIA Nsight Systems](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)。

MoE 模型和 CPU-GPU 同步開銷

Mixture-of-Experts (MoE) 模型的某些實現需要在每次迭代中進行 CPU-GPU 同步,以為選定的專家排程適當的 kernels。這引入了 kernel 啟動開銷,可能會顯著影響輸送量,尤其是當 CPU-GPU 連接速度慢時。例如,在 [MakoGenerate's optimization of DeepSeek MOE kernels](https://www.mako.ai/blog/mako-generate-achieves-1-83x-performance-over-torch-compile-on-deepseek-moe-kernels) 中,參考實現在每次前向傳遞中派發了 1,043 個 kernels,有 67 個 CPU-GPU 同步點。透過重構專家路由機制,他們將其減少到 533 個 kernel 啟動和僅 3 個同步點,實現了同步開銷減少 97%,端到端延遲減少 44%。請注意,並非所有 MoE 實現都需要 CPU-GPU 同步(現代實現通常將路由完全保留在 GPU 上),但對於那些需要的,高效的 CPU-GPU 通訊對於效能變得至關重要。

Grace Hopper Superchips:CPU-GPU 通訊的不同方法

與傳統的 x86+Hopper 系統相比,NVIDIA 的 Grace Hopper superchips 對 CPU-GPU 通訊採取了根本不同的方法。主要改進包括:
  - **1:1 GPU 對 CPU 比率**(相比 x86+Hopper 的 4:1),每個 GPU 提供 3.5 倍更高的 CPU 記憶體頻寬
  - **NVLink-C2C** 取代 PCIe Gen5 通道,提供 900 GB/s vs 128 GB/s(GPU-CPU 連結頻寬高 7 倍)
  - **NVLink Switch System** 提供比透過 PCIe Gen4 連接的 InfiniBand NDR400 NICs 高 9 倍的 GPU-GPU 連結頻寬

有關更多詳細資訊,請參閱 [NVIDIA Grace Hopper Superchip Architecture Whitepaper](https://download.deltacomputer.com/NVIDIA%20Grace%20Hopper%20Superchip%20Architecture%20Whitepaper.pdf)(第 11 頁)。

**⚠️ NUMA 親和性:對多插座效能至關重要**

在像我們的 AMD EPYC 7R13 節點這樣的多插座系統上(2 個插座,每個 48 核),****NUMA 親和性對 GPU 效能至關重要**。它指的是在與其目標裝置(如 GPU)共享相同插座的 CPU 核心上執行程序。當你的 GPU 程序在與 GPU 附加位置不同的 NUMA 節點的 CPU 上執行時,運算必須穿越 CPU 互連(AMD Infinity Fabric),增加顯著的延遲和頻寬約束。

**首先,讓我們檢查 NUMA 拓撲和節點距離以理解效能影響**:

```

$ numactl --hardware
node distances:
node   0   1
  0:  10  32
  1:  32  10

```

距離值顯示,在同一 NUMA 節點上存取記憶體(距離 10)比跨越到另一個 NUMA 節點(距離 32)快得多。這個記憶體存取**延遲**的 **3.2 倍差異**可能會在你的程序被固定到錯誤的 NUMA 節點時顯著影響 GPU 效能。

有關診斷和解決 NUMA 相關效能問題的詳細步驟,請參閱「故障排除互連效能」部分。

#### [GPU-to-GPU Intranode](https://huggingfacetb-smol-training-playbook.hf.space/#gpu-to-gpu-intranode)

在分散式訓練中,GPU 必須頻繁交換梯度、權重和啟動,通常每次迭代有數 GB 的資料。這大量的資料需要謹慎處理通訊。雖然 H100 的內部 HBM 可以以約 3 TB/s 的速度讀取,但意外使用錯誤的標誌可能會完全破壞你的 GPU-to-GPU 通訊頻寬!

讓我們透過檢查在同一節點內的 GPU 之間進行通訊的所有方式(以及你應該或不應該設定的所有標誌)來了解原因 🙂

**TL;DR:** 節點內的 GPU 可以透過三種方式通訊:透過 CPU(最慢,~3 GB/s,受 PCIe 瓶頸限制)、透過 EFA NICs 的 GPUDirect RDMA(~38 GB/s),或透過 NVLink 的 GPUDirect RDMA(~786 GB/s 雙向)。NVLink 快 9-112 倍,並完全繞過 CPU/PCIe。NCCL 在可用時自動優先使用 NVLink。NVLink SHARP (NVLS) 提供硬體加速的集體運算,將 allreduce 效能提升 1.3 倍至 480 GB/s。然而,alltoall 運算(340 GB/s)不受益於 NVLS 加速。

#### [**透過 CPU**](https://huggingfacetb-smol-training-playbook.hf.space/#through-cpu)

天真的方法使用 host memory (SHM):資料從 GPU1 經過 PCIe switch 到 CPU,進入 host memory,返回 CPU,再次經過 PCIe switch,最後到 GPU2。這可以透過 NCCL 使用 `NCCL_P2P_DISABLE=1` 和 `FI_PROVIDER=tcp` 環境變數來實現(儘管不推薦)。當這種模式被啟動時,你可以透過設定 `NCCL_DEBUG=INFO` 來驗證它,這將顯示如下訊息:

```

NCCL INFO Channel 00 : 1[1] -> 0[0] via SHM/direct/direct

```

CPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitch

EFA Link - 12.5 GB/sPCIe Gen4 - 16 GB/sPCIe Gen5 - 64 GB/sNVLink 4.0 - 900 GB/s

Bandwidth Max
for GPU ⟷ GPU via CPU
16
GB/s
-
Efficiency

Real Bandwidths

透過 CPU 和主記憶體的 GPU-to-GPU 通訊路徑,顯示透過 PCIe switch 和 CPU 的低效率往返。

這種迂迴路徑涉及多次記憶體複製並飽和 PCIe 和 CPU 記憶體匯流排,造成擁塞。在我們的拓撲中,4 個 H100 共享相同的 CPU 記憶體匯流排,當多個 GPU 嘗試同時通訊時,這種擁塞變得更加嚴重,因為它們競爭相同的有限 CPU 記憶體頻寬… 😢

使用這種 CPU 中介方法,我們從根本上受到 CPU 和 PCIe switch 之間的 PCIe Gen4 x8 連結約~16 GB/s 的瓶頸限制。幸運的是,我們的 GPU 有更好的通訊方式,無需涉及 CPU:**GPUDirect RDMA**。

#### [透過 Libfabric EFA](https://huggingfacetb-smol-training-playbook.hf.space/#through-libfabric-efa)

**GPUDirect RDMA**(Remote Direct Memory Access 或 GDRDMA)是一種技術,透過允許直接存取 GPU 記憶體來實現 NVIDIA GPU 之間的直接通訊。這消除了資料需要經過系統 CPU 的需求,並避免了透過系統記憶體的緩衝區複製,與傳統的 CPU 中介傳輸相比,效能提升高達 10 倍。GPUDirect RDMA 透過 PCIe 工作,以實現節點內(如我們在這裡看到的)和跨節點使用具有 RDMA 能力的 **NICs**(網路介面卡,我們將在未來的部分看到)的快速 GPU-to-GPU 通訊。有關更多詳細資訊,請參閱 [NVIDIA GPUDirect](https://developer.nvidia.com/gpudirect)。

回顧我們的拓撲圖,我們可以看到每個 **PCIe switch** 有 4 個 EFA (Elastic Fabric Adapter) NICs,這意味著每個 GPU 可以存取 4 個 EFA 適配器。**EFA** 是 AWS 的自訂高效能網路介面,用於雲端 instances,設計用於提供低延遲、高輸送量的 instance 間通訊。在 p5 instances 上,EFA 公開了一個 **libfabric** 介面(一個用於高效能計算的特定通訊 API),應用程式可以使用,並提供類似 RDMA 的能力,實現跨節點的直接 GPU-to-GPU 通訊的 **GPUDirect RDMA**。

EFA 使用 [稱為 Scalable Reliable Datagram (SRD) 的可靠基於乙太網的傳輸協議](https://ieeexplore.ieee.org/document/9167399),設計用於使用商品資料中心網路(具有大量網路路徑)。了解其重要性 [這裡](https://aws.amazon.com/blogs/hpc/in-the-search-for-performance-theres-more-than-one-way-to-build-a-network/)。

```

$ lstopo -v
...

## We can see 4 such EFA devices per each PCIe switch

PCIBridge L#8 (busid=0000:46:01.0 id=1d0f:0200 class=0604(PCIBridge) link=15.75GB/s buses=0000:[4f-4f] PCIVendor="Amazon.com, Inc.")
PCI L#6 (busid=0000:4f:00.0 id=1d0f:efa1 class=0200(Ethernet) link=15.75GB/s PCISlot=82-1 PCIVendor="Amazon.com, Inc.")
    OpenFabrics L#4 (NodeGUID=cd77:f833:0000:1001 SysImageGUID=0000:0000:0000:0000 Port1State=4 Port1LID=0x0 Port1LMC=1 Port1GID0=fe80:0000:0000:0000:14b0:33ff:fef8:77cd) "rdmap79s0"
...
$ fi_info --verbose
        fi_link_attr:
            address: EFA-fe80::14b0:33ff:fef8:77cd
            mtu: 8760            # maximum packet size is 8760 bytes

            speed: 100000000000  # each EFA link provides 100 Gbps of bandwidth

            state: FI_LINK_UP
            network_type: Ethernet

```

每個 **EFA link** 提供 100 Gbps (12.5 GB/s) 的頻寬。每個 GPU 有 **4 個 EFA NICs**,每個節點有 8 個 GPU,這提供了 100 × 4 × 8 = **3200 Gbps per node** (400GB/s) 的聚合頻寬。

有關如何使用 libfabric 和 EFA 充分利用這 3200 Gbps 頻寬的詳細探索,請參閱 Lequn Chen 的出色部落格系列:[Harnessing 3200 Gbps Network: A Journey with RDMA, EFA, and libfabric](https://le.qun.ch/en/blog/2024/12/25/libfabric-efa-0-intro/)。

為了確保我們啟用了透過 EFA 的 GPUDirect RDMA,你應該設定 `FI_PROVIDER=efa` 和 `NCCL_P2P_DISABLE=1` 環境變數。當這種模式被啟動時,你可以透過設定 `NCCL_DEBUG=INFO` 來驗證它是否正常工作,這將顯示如下訊息:

```

NCCL INFO Channel 01/1 : 1[1] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA/Shared

```

NodeCPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitchCPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitchNUMA 0NUMA 1

EFA Link - 12.5 GB/sPCIe Gen4 - 16 GB/sPCIe Gen5 - 64 GB/sNVLink 4.0 - 900 GB/s

Bandwidth Max
for GPU ⟷ GPU via EFA
50
GB/s
-
Efficiency

Real Bandwidths

透過 Libfabric EFA 的 GPU-to-GPU 通訊路徑。請注意,與使用 NVLink 相比,這對於節點內通訊效率較低。

雖然透過 EFA 的 GPUDirect RDMA 比 CPU 中介傳輸提供了顯著改進,每個 GPU 有 4 個 EFA 卡實現約 **50 GB/s**,我們能做得更好嗎?這就是 NVLink 發揮作用的地方。

#### [透過 NVLink](https://huggingfacetb-smol-training-playbook.hf.space/#through-nvlink)

**NVLink** 是 NVIDIA 的高速、直接 GPU-to-GPU 互連技術,實現伺服器內的快速多 GPU 通訊。**H100** 採用第四代 NVLink (NVLink 4.0),透過 18 個連結提供每個 GPU 900 GB/s 的雙向頻寬,每個連結以 50 GB/s 雙向運作([NVIDIA H100 Tensor Core GPU Datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c))。

在 **DGX H100** 架構中,4 個第三代 NVSwitches 使用分層拓撲連接 8 個 GPU,其中每個 GPU 跨 switches 以 5+4+4+5 連結連接。這種配置確保任何 GPU 對之間有多條直接路徑,常數跳數僅為 1 個 NVSwitch,產生 3.6 TB/s 總雙向 NVLink 網路頻寬。

NVLink 2.0 (Volta) | NVLink 3.0 (Ampere) | NVLink 4.0 (Hopper) | NVLink 5.0 (Blackwell)
---|---|---|---
Bandwidth | 300 GB/s | 600 GB/s | 900 GB/s

_表:NVLink 頻寬跨世代比較,顯示理論規格_

預設情況下,NCCL 在可用時優先使用 NVLink 進行節點內 GPU 通訊,因為它提供同一機器上 GPU 之間最低的延遲和最高的頻寬路徑。然而,如果你沒有正確設定標誌,你可能會阻止使用 NVLink!😱

NVLink 實現直接 GPU-to-GPU 記憶體存取,無需涉及 CPU 或系統記憶體。當 NVLink 不可用時,NCCL 回退到透過 PCIe 的 GPUDirect P2P,或當跨插座 PCIe 傳輸次優時使用 Shared Memory (SHM) 傳輸。

要驗證正在使用 NVLink,設定 `NCCL_DEBUG=INFO` 並尋找如下訊息:

```

NCCL INFO Channel 00/1 : 0[0] -> 1[1] via P2P/CUMEM

```

**CUMEM** 表示對等運算使用 **CUDA 記憶體句柄 (cuMem API)**。[在這裡了解更多。](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#cumem-host-allocations)

下圖說明了使用 NVLink 時資料採取的直接路徑:

CPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitch

EFA Link - 12.5 GB/sPCIe Gen4 - 16 GB/sPCIe Gen5 - 64 GB/sNVLink 4.0 - 900 GB/s

Bandwidth Max
for GPU ⟷ GPU via NVSwitch
900
GB/s
-
Efficiency

Real Bandwidths

透過 NVLink 的 GPU-to-GPU 通訊路徑。

透過 **NVLink 4.0** 的理論頻寬 900 GB/s 與 **EFA** 的~50 GB/s 相比,我們期望節點內通訊有 18 倍的優勢。為了在實務中驗證這一點,我們執行了 [NCCL's SendRecv performance test](https://github.com/NVIDIA/nccl-tests/blob/master/src/sendrecv.cu) 來測量不同通訊路徑的實際頻寬:

```

$ FI_PROVIDER=XXX NCCL_P2P_DISABLE=X sendrecv_perf -b 8 -e 8G -f 2 -g 1 -c 1 -n 100

```

GPU-> GPU measured bandwidth with NCCL's SendRecv test (H100 GPUs, 1 Node, 2 GPUs)
Reset View
Legend

這毫無疑問地顯示了 NVLink 的效率有多高:它實現了 364.93 GB/s,而 EFA 的 38.16 GB/s(快 9 倍,或雙向 18 倍)和 CPU 基線的 3.24 GB/s(快 112.6 倍)。這些測量證實了為什麼 NCCL 優先使用 NVLink 進行節點內 GPU 通訊,但為了進一步檢查 NVLink 的效能,讓我們使用 `nvbandwidth` 測量所有 GPU 對之間的雙向頻寬,使用兩個方向的同時複製:

```

./nvbandwidth -t device_to_device_bidirectional_memcpy_write_ce -b <message_size> -i 5
memcpy CE GPU(row) <-> GPU(column) Total bandwidth (GB/s)
           0         1         2         3         4         5         6         7
 0       N/A    785.81    785.92    785.90    785.92    785.78    785.92    785.90
 1    785.83       N/A    785.87    785.83    785.98    785.90    786.05    785.94
 2    785.87    785.89       N/A    785.83    785.96    785.83    785.96    786.03
 3    785.89    785.85    785.90       N/A    785.96    785.89    785.90    785.96
 4    785.87    785.96    785.92    786.01       N/A    785.98    786.14    786.08
 5    785.81    785.92    785.85    785.89    785.89       N/A    786.10    786.03
 6    785.94    785.92    785.99    785.99    786.10    786.05       N/A    786.07
 7    785.94    786.07    785.99    786.01    786.05    786.05    786.14       N/A
SUM device_to_device_bidirectional_memcpy_write_ce_total 44013.06

```

測量的雙向頻寬 **786 GB/s** 代表 **NVLink 4.0** 理論 900 GB/s 規格的 85%。使用 NVLink 完全繞過了 CPU 瓶頸(對於 gpu-to-gpu 通訊)!

但這如何轉化為集體通訊模式?讓我們使用來自 NCCL tests 的 `all_reduce_perf` [基準測試](https://github.com/NVIDIA/nccl-tests/blob/master/src/all_reduce.cu)測量單個節點內的 `allreduce` 效能。

有關集體通訊模式的快速複習,請參閱 [UltraScale Playbook Appendix](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=ring_allreduce)。

```

$ ./all_reduce_perf -b 8 -e 16G -f 2 -g 1 -c 1 -n 100

```

NCCL's All-Reduce performance test intranode

有關全面的基準測試腳本和配置,請參閱 [AWS Distributed Training Samples](https://github.com/aws-samples/awsome-distributed-training/tree/main/micro-benchmarks/nccl-tests) 上的出色集合。

但等等…我們實現了 480 GB/s,這超過了 NVLink 4.0 的理論單向頻寬 450 GB/s 😮 這是什麼魔法,這怎麼可能?

深入研究文件,似乎答案在於 **NVLink SHARP (NVLS)**,NVIDIA 的硬體加速集體運算技術。它們為 H100 GPU 的單個節點上的 allreduce 運算提供了約 1.3 倍的加速!

![Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/image_2891384e-bcac-80e2-9cc5-c2c46c7ab39b.B9LkpQ-__ZcdxHc.webp)

有關 NVSwitch 如何實現這些硬體加速集體運算的技術詳細資訊,請參閱 [NVSwitch architecture presentation](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)。

它們能在其他地方也有幫助嗎?讓我們檢查 [alltoall performance](https://github.com/NVIDIA/nccl-tests/blob/master/src/alltoall.cu):

```

$ ./all_to_all_perf -b 8 -e 16G -f 2 -g 1 -c 1 -n 100

```

NCCL's All-to-All performance test intranode

我們為 alltoall 運算實現了 **340 GB/s**,這與已發布的基準測試一致,顯示 H100 系統與 NVLink 4.0 的類似效能特性([來源](https://juser.fz-juelich.de/record/1019178/files/02-NCCL_NVSHMEM.pdf#page=20.00))。與 allreduce 不同,alltoall 運算不受益於 NVLS 硬體加速,這解釋了為什麼我們在這裡看到 340 GB/s,而 allreduce 實現了 480 GB/s。alltoall 模式需要所有 GPU 對之間更複雜的點對點資料交換,純粹依賴 NVLink 的基本頻寬而不是 NVSwitch 的集體加速功能。

對於自訂 NVLink 通訊模式,請關注 PyTorch 的 [SymmetricMemory API](https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798),它實現對 NVLink 和 NVLS 運算的細粒度控制。

進階 Kernel 優化

一些優化的 kernels 透過分配專用 warps 來處理傳輸,將 NVLink 通訊與計算分離。例如,ThunderKittens 使用 warp 層級設計,其中特定 warps 發出 NVLink 傳輸並等待完成,而其他 warps 繼續計算運算。這種 SM 計算和 NVLink 通訊的細粒度重疊可以隱藏大部分 GPU 間通訊延遲。有關實現詳細資訊,請參閱 [ThunderKittens blog post on multi-GPU kernels](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl#fine-grained-overlap-of-sm-compute-and-nvlink-communication-with-thunderkittens)。

雖然 NVLink 在單個節點內提供了出色的頻寬,但訓練前沿模型需要跨多個節點擴展。

這引入了一個新的潛在瓶頸:節點間網路互連,其運作頻寬顯著低於 NVLink。

#### [GPU-to-GPU Internode](https://huggingfacetb-smol-training-playbook.hf.space/#gpu-to-gpu-internode)

**TL;DR** 多節點 GPU 通訊使用高速網路,如 InfiniBand (400 Gbps) 或 RoCE (100 Gbps)。Allreduce 擴展良好(跨節點穩定在 320-350 GB/s),實現大規模訓練叢集。由於演算法複雜性,Alltoall 降級更明顯。延遲從節點內的~13μs 跳到節點間的 55μs+。對於需要頻繁 all-to-all 運算的 MoE 工作負載,NVSHMEM 提供非同步 GPU 發起的通訊,效能顯著優於 CPU 協調的傳輸。

隨著模型擴展超出單個節點可以容納的範圍,訓練需要透過高速網路連接的多個節點分散計算。在深入基準測試之前,讓我們看看你在多節點 GPU 叢集中會遇到的 3 種關鍵網路技術:
  - **Ethernet** 已從 1 Gbps 演變到 100+ Gbps 速度,並在 HPC 和資料中心叢集中廣泛使用。
  - **RoCE (RDMA over Converged Ethernet)** 將 RDMA 能力帶到乙太網路,使用 ECN 進行擁塞控制,而不是傳統的 TCP 機制。
  - **InfiniBand** 是 NVIDIA 的行業標準 switch fabric,提供高達 400 Gbps 的頻寬和次微秒延遲,支援 RDMA,透過 GPUDirect RDMA 繞過 host CPU 實現直接 GPU-to-GPU 記憶體存取。

總結如下:

Name | Ethernet (25–100 Gbps) | Ethernet (200–400 Gbps) | RoCE | Infiniband
---|---|---|---|---
Manufacturer | Many | Many | Many | NVIDIA/Mellanox
Unidirectional Bandwidth (Gbps) | 25–100 | 200–400 | 100 | 400
End to End Latency (μs) | 10-30 | N/A | ~1 | <1
RDMA | No | No | Yes | Yes

表:互連比較。來源:<https://www.sciencedirect.com/science/article/pii/S2772485922000618>

對於 AWS p5 instances,我們有 [Elastic Fabric Adapter (](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)[**EFA**](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)[)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) 作為 **NIC**(Network Interface Card),如我們之前看到的,每個 GPU 透過 PCIe Gen5 x16 通道連接到四個 100 Gbps EFA 網路卡。

NodeNUMA 1...CPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitchNUMA 0NodeNUMA 1...CPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitchNUMA 0

EFA Link - 12.5 GB/sPCIe Gen4 - 16 GB/sPCIe Gen5 - 64 GB/sNVLink 4.0 - 900 GB/s

Bandwidth Max
for GPU ⟷ GPU via EFA
50
GB/s
-
Efficiency

Real Bandwidths

透過 Libfabric EFA 的節點間 GPU-to-GPU 通訊路徑

如上圖所示,當 GPU 和網路卡連接到同一個 PCIe switch 時,**GPUDirect RDMA** 使它們的通訊僅透過該 switch 發生。這種設定允許充分利用 PCIe Gen5 x16 頻寬,並避免涉及其他 PCIe switches 或 CPU 記憶體匯流排。理論上,每個節點 8 個 PCIe Switches x 每個 switch 4 個 EFA NICs x 每個 EFA NIC 100 Gbps 提供 **3200 Gbps**(400GB/s)** 的頻寬,這是我們在 [AWS p5's specs](https://aws.amazon.com/ec2/instance-types/p5/) 中找到的頻寬。那麼它在實務中如何表現?讓我們透過執行與之前相同的基準測試但跨不同節點來找出答案!

**頻寬分析**

NCCL's Sendrecv performance test
NCCL's All-Reduce performance test
NCCL's Alltoall performance test

Number of Nodes

跨不同節點數的集體運算頻寬擴展,在我們的 AWS p5 instances 上,使用來自 [aws-samples/awsome-distributed-training](https://github.com/aws-samples/awsome-distributed-training/blob/main/micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch) 的建議。

點對點 send/receive 運算對於 2-4 個節點實現約 **42-43 GB/s**,但對於 5+ 個節點降至約 21 GB/s。這種效能降級發生是因為 NCCL 在擴展超過 4 個節點時自動將每個對等的點對點通道數從 2 減少到 1,有效地將可用頻寬利用率減半,而理論最大值保持在~50 GB/s(4 個 EFA NICs × 12.5 GB/s each)。我們成功地透過設定 `NCCL_NCHANNELS_PER_NET_PEER=2` 恢復了 5+ 個節點上此測試的全輸送量,儘管應謹慎使用此標誌,因為它可能會降低 all-to-all 效能,例如(有關詳細資訊,請參閱 [GitHub issue #1272](https://github.com/NVIDIA/nccl/issues/1272))。

all-reduce 運算在單個節點內表現出色,實現了 **480 GB/s** 的匯流排頻寬。當擴展到 2 個節點時,頻寬幾乎保持不變,為 479 GB/s,之後穩定在 3-16 個節點的約 320-350 GB/s。這種模式揭示了一個重要特性:雖然由於從 NVLink 到節點間網路結構的轉換,跨越節點邊界時存在初始下降,但_當我們添加更多節點時,頻寬幾乎保持恆定擴展。_

跨節點擴展 All-Reduce

這種超過 2 個節點的近乎恆定擴展行為實際上對大規模訓練來說是相當令人鼓舞的。跨 3-16 個節點相對穩定的 320-350 GB/s 表明,依賴 all-reduce 運算的並行策略(例如,在資料並行中)可以擴展到數百甚至數千個 GPU,而不會顯著降低每個 GPU 的頻寬。這種對數擴展特性是使用 8-rail 優化 fat trees 的精心設計的多層網路拓撲的典型特徵,其中 8 個 GPU 中的每一個都連接到單獨的 switch rail 以最大化平分頻寬。現代前沿訓練叢集常規運作在 100,000+ GPU,這種穩定的擴展行為使如此大規模的部署成為可能。

當處理不同的頻寬連結(節點內的 NVLink vs. 節點間網路)時,考慮調整你的並行策略到每個頻寬層級以充分利用所有可用頻寬。有關優化異構網路拓撲的並行配置的詳細指導,請參閱 [Ultrascale playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。

all-to-all 運算顯示出更戲劇性的擴展挑戰:從單個節點的 344 GB/s 開始,頻寬在 2 個節點時降至 81 GB/s,並繼續下降到較大叢集的約 45-58 GB/s。這種更陡峭的降級反映了 all-to-all 模式的密集網路需求,其中每個 GPU 必須跨節點與其他每個 GPU 通訊,比 all-reduce 運算創造了顯著更多的網路擁塞。

**延遲分析**

NCCL's Sendrecv performance test
8 B128 B2 KB32 KB512 KB8 MB128 MB2 GB8 GB50 μs100 μs200 μs500 μs1.0 ms2.0 ms5.0 ms10.0 ms20.0 ms50.0 ms100.0 ms200.0 ms500.0 msLatency (μs)Message Size (bytes)

NCCL's All-Reduce performance test
8 B128 B2 KB32 KB512 KB8 MB128 MB2 GB8 GB50 μs100 μs200 μs500 μs1.0 ms20 μs2.0 ms5.0 ms10.0 ms20.0 ms50.0 msLatency (μs)Message Size (bytes)

NCCL's Alltoall performance test
128 B2 KB32 KB512 KB8 MB128 MB2 GB8 GB50 μs100 μs200 μs500 μs1.0 ms10 μs20 μs2.0 ms5.0 ms10.0 ms20.0 ms50.0 ms100.0 ms200.0 msLatency (μs)Message Size (bytes)

Number of Nodes
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

跨不同節點數的集體運算延遲擴展,在我們的 AWS p5 instances 上,使用來自 [[aws-samples/awsome-distributed-training](https://github.com/aws-samples/awsome-distributed-training/blob/main/micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch)](https://github.com/aws-samples/awsome-distributed-training/blob/main/micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch) 的建議。

**延遲**測量揭示了跨越節點邊界的基本成本。Send/receive 運算在所有多節點配置中保持相對穩定的延遲 **40-53 μs**,表明點對點通訊延遲主要由基本網路往返時間決定,而不是叢集大小,儘管一些變化表明網路拓撲和路由效果仍然發揮作用。

All-reduce 運算在單個節點內顯示 **12.9 μs** 的最小延遲,但對於 2 個節點跳到 **55.5 μs**,並隨著叢集大小幾乎線性增加,在 16 個節點時達到 **235 μs**。這種進展反映了增加的通訊距離和跨更多節點的 reduction tree 的增長複雜性。

All-to-all 運算表現出類似的趨勢,單節點通訊從 **7.6 μs** 開始,但在 2 個節點時攀升到 **60 μs**,在 16 個節點時達到 **621** μs。all-to-all 運算延遲的超線性增長表明,隨著更多節點參與集體,網路擁塞和協調開銷會加劇。

NVSHMEM 用於優化 GPU 通訊

隨著 Mixture of Experts (MoE) 架構的興起,需要頻繁的 all-to-all 通訊模式進行專家路由,優化的 GPU 通訊庫變得越來越重要。

[NVSHMEM](https://developer.nvidia.com/nvshmem) 作為高效能通訊庫獲得了顯著的關注,它將多個 GPU 的記憶體組合成分區全域地址空間 (PGAS)。與依賴 CPU 協調資料傳輸的傳統基於 MPI 的方法不同,NVSHMEM 實現非同步、GPU 發起的運算,消除 CPU-GPU 同步開銷。

NVSHMEM 為 GPU 通訊提供了幾個關鍵優勢:透過 GPUDirect Async 等技術,GPU 可以在發出節點間通訊時完全繞過 CPU,對於小訊息(<1 KiB)實現高達 9.5 倍的輸送量。這對於需要密集網路通訊模式的集體運算特別有益。

該庫目前支援 InfiniBand/RoCE 與 Mellanox 適配器(CX-4 或更高版本)、Slingshot-11 (Libfabric CXI) 和 Amazon EFA (Libfabric EFA)。對於需要細粒度通訊的強擴展應用,與傳統的 CPU-proxy 方法相比,NVSHMEM 的低開銷、單側通訊原語可以顯著提高效能。

在 [NVSHMEM documentation](https://developer.nvidia.com/nvshmem) 和這篇詳細的 [blog post on GPUDirect Async](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/) 中了解更多。

當頻寬測量低於預期時,幾個因素可能會限制效能。理解這些潛在的瓶頸對於實現最佳互連利用率至關重要。

#### [故障排除互連](https://huggingfacetb-smol-training-playbook.hf.space/#troubleshooting-interconnect)

如果你遇到低於預期的頻寬,請系統地檢查以下區域:

**庫版本**

過時的 NCCL、EFA 或 CUDA 庫可能缺少關鍵的效能優化或錯誤修復。始終驗證你正在執行所有通訊庫的最新、相容版本。例如,AWS 定期更新其 Deep Learning AMIs,為其硬體提供優化的庫版本。還建議為重要實驗記錄這些庫版本。

**CPU 親和性配置**

不適當的 CPU 親和性設定可能會透過引起不必要的跨 NUMA 流量顯著影響 NCCL 效能。每個 GPU 應該綁定到同一 NUMA 節點上的 CPU,以最小化記憶體存取延遲。實際上,[this Github issue](https://github.com/NVIDIA/nccl/issues/1017#issuecomment-1751385723) 展示了使用 `NCCL_IGNORE_CPU_AFFINITY=1` 和 `--cpu-bind none` 如何幫助顯著減少容器延遲。[你可以在這裡閱讀更多。](https://enterprise-support.nvidia.com/s/article/understanding-numa-node-for-performance-benchmarks#Mapping-between-PCI-device-driver-port-and-NUMA)

**網路拓撲和放置**

理解你的網路拓撲對於診斷效能問題至關重要。雲端放置群組雖然有幫助,但不保證 instances 之間的最小網路跳數。在現代資料中心 fat-tree 拓撲中,放置在不同頂級 switches 下的 instances 將由於路由路徑中的額外網路跳數而經歷更高的延遲和潛在較低的頻寬。

對於 **AWS EC2** 使用者,[Instance Topology API](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-topology-works.html) 提供了有關網路節點放置的寶貴可見性。在底層(直接連接到 instance)共享相同網路節點的 Instances 物理上最接近,將實現最低延遲通訊。

```

graph TD
    %% Single Level 1 node
    L1["Level 1
nn-4aed5..."]:::level1

    %% Single Level 2 node
    L2["Level 2
nn-48b0a..."]:::level2

    %% 12 nodes of Level 3
    L3_1["Level 3
nn-d2ad4..."]:::level3
    L3_2["Level 3
nn-2d36a..."]:::level3
    L3_3["Level 3
nn-65fc9..."]:::level3
    L3_4["Level 3
nn-fbb73..."]:::level3
    L3_5["Level 3
nn-65290..."]:::level3
    L3_6["Level 3
nn-27373..."]:::level3
    L3_7["Level 3
nn-5adeb..."]:::level3
    L3_8["Level 3
nn-dbe4f..."]:::level3
    L3_9["Level 3
nn-fde84..."]:::level3
    L3_10["Level 3
nn-3c5c0..."]:::level3
    L3_11["Level 3
nn-94247..."]:::level3
    L3_12["Level 3
nn-8f3c1..."]:::level3

    %% L1 -> L2
    L1 --> L2

    %% L2 -> L3 (12 arrows)
    L2 --> L3_1
    L2 --> L3_2
    L2 --> L3_3
    L2 --> L3_4
    L2 --> L3_5
    L2 --> L3_6
    L2 --> L3_7
    L2 --> L3_8
    L2 --> L3_9
    L2 --> L3_10
    L2 --> L3_11
    L2 --> L3_12

    %% Distribution Level 3 -> Leaf nodes (instance info)
    %% 1st Level 3 has 2 leaves
    L3_1 --> L4_1["ID: 02e1b4f9
ip-26-0-171-102
p5.48xlarge"]:::level4
    L3_1 --> L4_2["ID: 05388ebf
ip-26-0-171-230
p5.48xlarge"]:::level4

    %% 2nd, 3rd, 4th have 1 each
    L3_2 --> L4_3["ID: 03bfac00
ip-26-0-168-30
p5.48xlarge"]:::level4
    L3_3 --> L4_4["ID: d92bab46
ip-26-0-168-95
p5.48xlarge"]:::level4
    L3_4 --> L4_5["ID: 97a542e4
ip-26-0-163-158
p5.48xlarge"]:::level4

    %% 5th has 3
    L3_5 --> L4_6["ID: e2c87e43
ip-26-0-167-9
p5.48xlarge"]:::level4
    L3_5 --> L4_7["ID: afa887ea
ip-26-0-168-120
p5.48xlarge"]:::level4
    L3_5 --> L4_8["ID: 66c12e70
ip-26-0-167-177
p5.48xlarge"]:::level4

    %% 6th, 7th, 8th have 1 each
    L3_6 --> L4_9["ID: 9412bdf3
ip-26-0-168-52
p5.48xlarge"]:::level4
    L3_7 --> L4_10["ID: 87bd4dc8
ip-26-0-167-111
p5.48xlarge"]:::level4
    L3_8 --> L4_11["ID: b001549b
ip-26-0-166-244
p5.48xlarge"]:::level4

    %% 9th has 2
    L3_9 --> L4_12["ID: 10ed8172
ip-26-0-107-245
p5.48xlarge"]:::level4
    L3_9 --> L4_13["ID: 7c1d0a09
ip-26-0-168-238
p5.48xlarge"]:::level4

    %% 10th, 11th, 12th have 1 each
    L3_10 --> L4_14["ID: 925ce932
ip-26-0-167-217
p5.48xlarge"]:::level4
    L3_11 --> L4_15["ID: c9bc34db
ip-26-0-171-168
p5.48xlarge"]:::level4
    L3_12 --> L4_16["ID: 328d5d04
ip-26-0-167-127
p5.48xlarge"]:::level4

    %% Styles
    classDef level1 fill:#c8e6c9
    classDef level2 fill:#e1f5fe
    classDef level3 fill:#fff9c4
    classDef level4 fill:#ffcdd2

```

網路拓撲視覺化顯示 instance 放置。

最小化通訊節點之間的網路跳數直接轉化為更好的互連效能。對於小規模實驗和消融,確保你的 instances 共同位於同一網路 switch 上可以在延遲和頻寬利用率方面產生可測量的差異。

**正確的環境變數**

缺少或不正確的網路適配器環境變數可能會嚴重限制頻寬利用率。像 NCCL 這樣的通訊庫依賴特定的配置標誌來啟用最佳效能功能,如自適應路由、GPU 發起的傳輸和適當的緩衝區大小。

例如,當使用 AWS EFA (Elastic Fabric Adapter) 時,確保你為你的 instance 類型設定了推薦的 NCCL 和 EFA 環境變數。[AWS EFA cheatsheet](https://github.com/aws-samples/awsome-distributed-training/blob/main/1.architectures/efa-cheatsheet.md) 為不同場景提供了關於最佳標誌配置的全面指導。

**容器特定考慮因素**

使用容器(Docker/Enroot)時,幾個配置步驟對於最佳 NCCL 效能至關重要:
  - **Shared 和 Pinned Memory**:Docker 容器預設限制 shared 和 pinned memory 資源。使用 `-shm-size=1g --ulimit memlock=-1` 啟動容器以防止初始化失敗。
  - **NUMA 支援**:Docker 預設禁用 NUMA 支援,這可能會阻止 cuMem host allocations 正常工作。透過使用 `-cap-add SYS_NICE` 調用 Docker 來啟用 NUMA 支援。
  - **PCI 拓撲發現**:確保 `/sys` 正確掛載,以允許 NCCL 發現 GPU 和網路卡的 PCI 拓撲。讓 `/sys` 公開虛擬 PCI 拓撲可能導致次優效能。

社群故障排除

我們正在作為社群努力在這裡收集故障排除發現。如果你遇到了效能問題或發現了有效的除錯方法,請跳到 [Discussion Tab](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook/discussions) 並分享你的經驗,以幫助其他人優化他們的互連利用率。

現在你知道如何除錯 GPU-CPU 和 GPU-GPU 通訊中的瓶頸,讓我們看看通常較少受到關注的 GPU 通訊部分,即與儲存層的通訊!

#### [GPU-to-Storage](https://huggingfacetb-smol-training-playbook.hf.space/#gpu-to-storage)

GPU 和儲存系統之間的連接通常被忽視,但可能會顯著影響訓練效率。在訓練期間,GPU 需要不斷從儲存讀取資料(資料載入,特別是對於具有大型影像/視訊檔案的多模態資料),並定期將模型狀態寫回儲存(即 checkpointing)。對於現代大規模訓練執行,如果未正確優化,這些 I/O 運算可能成為瓶頸。

**TL;DR:** GPU-storage I/O 透過資料載入和 checkpointing 影響訓練。GPUDirect Storage (GDS) 實現直接 GPU-to-storage 傳輸,繞過 CPU 以獲得更好的效能。即使在我們的叢集中未啟用 GDS,本地 NVMe RAID(RAID 0 中的 8×3.5TB 硬碟)也提供 26.59 GiB/s 和 337K IOPS(比網路儲存快 6.3 倍),使其成為 checkpoints 的理想選擇。

**理解儲存拓撲**

GPU 和儲存裝置之間的物理連接遵循類似於 GPU 互連的階層結構。儲存裝置透過 PCIe bridges 連接,理解這種拓撲有助於解釋效能特性和潛在的瓶頸。

從 `lstopo` 查看系統拓撲,我們可以看到 NVMe 硬碟如何連接到系統。在我們的 p5 instance 中,我們每個 GPU 有 1 個 NVMe SSD:

```

PCIBridge L#13 (busid=0000:46:01.5 id=1d0f:0200 class=0604(PCIBridge) link=15.75GB/s buses=0000:[54-54] PCIVendor="Amazon.com, Inc.")
PCI L#11 (busid=0000:54:00.0 id=1d0f:cd01 class=0108(NVMExp) link=15.75GB/s PCISlot=87-1 PCIVendor="Amazon.com, Inc." PCIDevice="NVMe SSD Controller")
    Block(Disk) L#9 (Size=3710937500 SectorSize=512 LinuxDeviceID=259:2 Model="Amazon EC2 NVMe Instance Storage" Revision=0 SerialNumber=AWS110C9F44F9A530351) "nvme1n1"

```

一個自然的問題是 GPU 是否可以直接存取 NVMe 硬碟而不涉及 CPU。答案是肯定的,透過 **GPUDirect Storage (GDS)**。

**GPUDirect Storage** 是 NVIDIA 的 [GPUDirect](https://developer.nvidia.com/gpudirect) 技術家族的一部分,實現儲存(本地 NVMe 或遠端 NVMe-oF)和 GPU 記憶體之間的直接資料路徑。它透過允許儲存控制器附近的 DMA 引擎直接將資料移入或移出 GPU 記憶體,消除了透過 CPU bounce buffers 的不必要記憶體複製。這減少了 CPU 開銷,降低了延遲,並顯著提高了資料密集型工作負載(如在大型多模態資料集上訓練)的 I/O 效能。

要驗證系統上是否正確配置了 GPUDirect Storage,你可以檢查 GDS 配置檔案並使用提供的診斷工具:

```

$ /usr/local/cuda/gds/tools/gdscheck.py -p
 =====================
 DRIVER CONFIGURATION:
 =====================
 NVMe               : Supported   NVMeOF             : Unsupported
 SCSI               : Unsupported
 ScaleFlux CSD      : Unsupported
 NVMesh             : Unsupported
 DDN EXAScaler      : Unsupported
 IBM Spectrum Scale : Unsupported
 NFS                : Unsupported
 BeeGFS             : Unsupported
 WekaFS             : Unsupported
 Userspace RDMA     : Unsupported
 --Mellanox PeerDirect : Enabled
 --rdma library        : Not Loaded (libcufile_rdma.so)
 --rdma devices        : Not configured
 --rdma_device_status  : Up: 0 Down: 0
 =====================

```

我們看到 `NVMe: Supported`,這表明 GDS 目前配置為適用於 NVMe 硬碟,所有其他儲存類型都未正確配置,如 Unsupported 標誌所示。如果 GDS 未正確配置為你的儲存類型,請參閱 [NVIDIA GPUDirect Storage Configuration Guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html) 以獲取有關修改 `/etc/cufile.json` 配置檔案的說明。

**區塊儲存裝置**

要理解系統上可用的儲存裝置,你可以使用 `lsblk` 顯示區塊裝置層次結構:

```

$ lsblk --fs -M
    NAME        FSTYPE            LABEL                   UUID                                 FSAVAIL FSUSE% MOUNTPOINT
...
    nvme0n1
    └─nvme0n1p1 ext4              cloudimg-rootfs         24ec7991-cb5c-4fab-99e5-52c45690ba30  189.7G    35% /
┌┈▶ nvme1n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
├┈▶ nvme2n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
├┈▶ nvme3n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
├┈▶ nvme8n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
├┈▶ nvme5n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
├┈▶ nvme4n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
├┈▶ nvme6n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
└┬▶ nvme7n1     linux_raid_member ip-26-0-164-236:MY_RAID d0795631-71f0-37e5-133b-e748befec126
 └┈┈md0         xfs                                       dddb6849-e5b5-4828-9034-96da65da27f0   27.5T     1% /scratch

```

此輸出顯示系統上的區塊裝置層次結構。主要觀察:
  - `nvme0n1p1` 是掛載在 `/` 的根 [Amazon EBS](https://aws.amazon.com/ebs/) 檔案系統,使用其全部~300GB 容量的 35%
  - 八個 NVMe 硬碟(`nvme1n1` 到 `nvme8n1`)配置為名為 `MY_RAID` 的 RAID 陣列
  - RAID 陣列公開為 `/dev/md0`,格式化為 XFS,並掛載在 `/scratch`,有 28TB 可用(8x3.5TB)

箭頭(┈▶)表示多個 NVMe 裝置是同一 RAID 陣列的成員,然後組合成單個 `md0` 裝置。

[Amazon Elastic Block Store (EBS)](https://aws.amazon.com/ebs/) 是一種高效能、可擴展的區塊儲存服務,設計用於與 Amazon EC2 instances 一起使用。

**網路儲存**

除了本地 NVMe 儲存,系統還可以存取網路附加儲存系統:

```

$ df -h
Filesystem                                         Size  Used Avail Use% Mounted on
/dev/root                                          291G  101G  190G  35% /
weka-hopper.hpc.internal.huggingface.tech/default  393T  263T  131T  67% /fsx
10.53.83.155@tcp:/fg7ntbev                         4.5T  2.9T  1.7T  63% /admin
/dev/md0                                            28T  206G   28T   1% /scratch

```

此輸出顯示:
  - `/dev/root`(291GB [Amazon EBS](https://aws.amazon.com/ebs/))是根檔案系統,容量為 35%
  - `/fsx`(393TB WekaFS)已滿 67%,有 131TB 可用
  - `/admin`(4.5TB FSx Lustre)已滿 63%,有 1.7TB 可用
  - `/dev/md0`(28TB 本地 NVMe RAID)僅滿 1%,在 `/scratch` 有 28TB 可用。這是我們在 RAID 中的 8×3.5TB SSD NVMe instance store 硬碟。

請注意,`/fsx` 實際上不是 Amazon FSx,而是 WekaFS。當我們從 FSx 遷移到 WekaFS 時,為了方便起見,我們保留了相同的掛載點名稱。

本地 NVMe RAID 陣列(`/scratch`)提供最快的 I/O 效能,而網路檔案系統為共享資料儲存提供更大的容量。

儲存技術

**RAID (Redundant Array of Independent Disks)**:組合多個硬碟以透過資料條帶化、同位檢查或鏡像來提高效能和/或可靠性。

**NVMe (Non-Volatile Memory Express)**:用於 SSD 的高效能儲存協議,直接連接到 PCIe,提供比 SATA/SAS 更高的輸送量和更低的延遲。

**WekaFS**:為 AI/ML 工作負載設計的高效能並行檔案系統,跨多個節點提供低延遲存取和高輸送量。

**FSx Lustre**:為 HPC 設計的並行檔案系統,在不同伺服器上分離 metadata 和資料服務以實現並行存取。雖然對大型檔案有效,但它可能在涉及許多小檔案的 metadata 密集型 AI/ML 工作負載中掙扎。

**基準測試儲存頻寬**

要理解每個儲存系統的效能特性,我們可以使用 GPUDirect Storage (GDS) 基準測試它們的讀/寫速度。這是一個測試各種配置的全面參數化基準測試腳本:

```

gdsio -f /<disk_path>/gds_test.dat -d 0 -w <n_threads> -s 10G -i <io_size> -x 1 -I 1 -T 10

```

基準測試評估儲存系統效能跨輸送量、延遲、IOPS,但也:

**可擴展性**:效能如何隨不同執行緒數和 I/O 大小變化。這揭示了不同工作負載模式的最佳配置:
  - 小 I/O 大小(64K 到 256K)通常最大化 IOPS,但可能不會飽和頻寬
  - 大 I/O 大小(2M 到 8M)通常最大化輸送量但減少 IOPS
  - 執行緒數影響兩者:更多執行緒可以增加總 IOPS 和輸送量,直到硬體限制

**傳輸方法效率**:比較 GPU_DIRECT vs CPU_GPU vs CPUONLY 顯示繞過 CPU 記憶體的好處:
  - **GPU_DIRECT**:使用 RDMA 直接將資料傳輸到 GPU 記憶體,完全繞過 CPU(最低延遲,最高效率,小運算的最佳 IOPS)
  - **CPU_GPU**:傳統路徑,資料首先進入 CPU 記憶體,然後複製到 GPU(增加 CPU 開銷和記憶體頻寬競爭,減少有效 IOPS)
  - **CPUONLY**:基線僅 CPU I/O,無 GPU 參與

IOPS (I/O Operations Per Second)

IOPS 是每秒完成的個別 I/O 運算數。從 gdsio 輸出計算為 `ops / total_time`。IOPS 對於以下情況特別重要:
  - 具有小 I/O 大小的隨機存取模式
  - 具有許多小檔案或分散資料存取的工作負載
  - 類似資料庫的運算,其中每個運算的延遲比原始頻寬更重要
  - 更高的 IOPS 表示更好的處理並行、細粒度資料存取的能力

/scratch - Bandwidth (GiB/s)
/root - Bandwidth (GiB/s)
/fsx - Bandwidth (GiB/s)
/admin - Bandwidth (GiB/s)

Transfer Type CPU ONLY GPU DIRECT CPU_GPU
Metric Bandwidth IOPS

跨不同執行緒數和 I/O 大小比較儲存系統效能的基準測試結果。熱圖視覺化輸送量(GiB/s)和 IOPS 模式,揭示每個儲存層級的最佳配置。注意:GPUDirect Storage (GDS) 目前在此叢集配置中不受支援。

基準測試揭示了我們四個儲存系統之間的巨大效能差異:

**/scratch (本地 NVMe RAID)** 以 **26.59 GiB/s** 輸送量和 **337K IOPS** 主導,使其比 FSx 快 6.3 倍(輸送量)和 6.6 倍(IOPS)。這個由 8×3.5TB NVMe 硬碟組成的本地 RAID 陣列提供最低延遲(峰值 IOPS 時為 190μs),並且隨執行緒數擴展異常好,在 64 個執行緒和 1M I/O 大小時實現峰值輸送量效能。

**/fsx (WekaFS)** 以 **4.21 GiB/s** 和 **51K IOPS** 提供穩定的網路儲存效能,使其成為需要合理效能的共享資料的最佳選擇。FSx 使用 CPUONLY 傳輸實現其最佳輸送量(4.21 GiB/s),而其最佳 IOPS (51K) 使用 GPUD 傳輸類型。

**/admin (FSx Lustre)** 和 **/root (EBS)** 檔案系統顯示類似的適度效能,約 **1.1 GiB/s** 輸送量,但在 IOPS 能力方面差異顯著。Admin 使用 GPUD 傳輸實現其峰值輸送量(1.13 GiB/s),並使用 CPU_GPU 傳輸達到 17K IOPS 峰值(比 Root 好 24 倍),使其更適合有許多小運算的工作負載。Root 的差 IOPS 效能(730)證實它最適合僅大型順序運算。

**關於 GPU_DIRECT 結果的注意事項:** GPUDirect Storage (GDS) 目前在我們的叢集中未啟用,這解釋了為什麼 NVMe 儲存(Scratch 和 Root)的 GPUD 結果與 CPUONLY 傳輸相比表現不佳。正確配置 GDS 後,我們期望 GPUD 對於直接 GPU-to-storage 傳輸顯示顯著優勢,特別是對於高效能 NVMe 陣列。

**最佳配置模式**:跨所有儲存類型,最大輸送量發生在 1M I/O 大小,而最大 IOPS 發生在最小測試大小(64K)。這種經典權衡意味著根據工作負載特性在原始頻寬(大 I/O)和運算並行(小 I/O)之間選擇。對於具有大型 checkpoint 檔案的 ML 訓練,Scratch 上的 1M-8M 範圍提供最佳效能。

#### [總結](https://huggingfacetb-smol-training-playbook.hf.space/#summary)

如果你已經走到這一步,恭喜!你現在對儲存層次結構以及不同元件如何在我們的訓練基礎設施中互動有了全面的理解。但這是我們希望你帶回家的關鍵見解:**識別瓶頸是將理論知識與實際優化區分開的因素**。

在整個指南中,我們測量了堆疊每個層級的實際頻寬:單個 GPU 內的 HBM3 為 3TB/s,節點內 GPU 之間的 NVLink 為 786 GB/s,CPU-GPU 傳輸的 PCIe Gen4 x8 為 14.2 GB/s,點對點通訊的節點間網路為 42 GB/s,以及從 26.59 GB/s(本地 NVMe)到 1.1 GB/s(共享檔案系統)的儲存系統。這些測量揭示了訓練管線將在哪裡減速,對於實現高 Model FLOPs Utilization (MFU) 至關重要。

然而,僅原始頻寬數字並不能說明完整的故事。現代訓練系統可以**重疊計算與通訊**,有效地將通訊成本隱藏在計算運算後面。這種並行化有助於緩解瓶頸,即使在互連速度慢時也是如此。有關重疊計算和通訊以最大化輸送量的詳細策略,請參閱 Ultra-Scale Playbook。

下面的圖表將我們所有基準測試的測量綜合到一個視圖中,顯示隨著我們遠離 GPU,頻寬如何顯著降低:

NodeCPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitchCPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUPCIeSwitchEFANVMeGPUNVSwitchNVSwitchNUMA 0NUMA 1

Show Real Bandwidths

Select path... Intranode: CPU ⟷ GPU Intranode: GPU ⟷ GPU via CPU Intranode: GPU ⟷ GPU via NVSwitch Intranode: GPU ⟷ GPU via EFA Internode: GPU ⟷ GPU via EFA Storage: GPU ⟷ Storage Storage: CPU ⟷ Storage Storage: GPU ⟷ Storage via CPU

EFA Link - 12.5 GB/sPCIe Gen4 - 16 GB/sPCIe Gen5 - 64 GB/sNVLink 4.0 - 900 GB/s

Real Bandwidth
for CPU → GPU
-
GB/s
-
Efficiency

Real Bandwidths

現在我們知道如何識別硬體和軟體設定中的瓶頸,讓我們看看如何進一步確保我們有一個彈性系統,可以穩定執行數月。
