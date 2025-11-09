### [GPU 內部:內部架構](https://huggingfacetb-smol-training-playbook.hf.space/#inside-a-gpu-internal-architecture)

GPU 從根本上來說是一個針對輸送量而非延遲進行優化的大規模並行處理器。與擅長快速執行少數複雜指令流的 CPU 不同,GPU 透過同時執行數千個簡單操作來實現效能。

理解 GPU 效能的關鍵在於認識到這不僅僅是關於原始計算能力,而是關於計算和資料移動之間的相互作用。一個 GPU 可能具有 teraflops 的理論計算能力,但如果資料無法足夠快地到達計算單元,那麼這種潛力就會被浪費。這就是為什麼我們需要同時理解記憶體層次結構(資料如何移動)和計算管線(工作如何完成)。

在最高層面上,GPU 因此執行兩項基本任務:
  1. **移動和儲存資料**(記憶體系統)
  2. **對資料進行有用的工作**(計算管線)

#### [計算單元和 FLOPs](https://huggingfacetb-smol-training-playbook.hf.space/#compute-units-and-flops)

**TL;DR:** GPU 以 FLOPs(每秒浮點運算)來衡量效能。現代 GPU 如 H100 在較低精度下提供顯著更高的輸送量:BF16 為 990 TFLOPs,而 FP32 為 67 TFLOPs。然而,由於記憶體瓶頸,實際效能是理論峰值的 70-77%。最先進的訓練達到 20-41% 的端到端效率,也稱為模型 flops 利用率 (MFU)。在規劃訓練執行時,使用實際數字,而不是行銷規格。

GPU 計算效能以 **FLOPs**(每秒浮點運算)來衡量。一個 FLOP 是單次算術運算,通常是浮點數加法,如 `a + b`,現代 GPU 每秒可以執行數兆次這樣的運算(TFLOPs)。

GPU 計算的基本建構單元是 **Streaming Multiprocessors (SMs)**,獨立的處理單元並行執行指令。每個 SM 包含兩種類型的**核心**: **CUDA cores** 用於標準浮點運算,以及專門的 **Tensor Cores** 優化用於矩陣乘法,這是深度學習中的主力運算(對 transformer 效能至關重要)。

現代 GPU 在晶片上組織了數百個這樣的 SM!例如,[H100](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) SXM5 版本(這是我們在叢集上使用的 GPU)包含 132 個 SM。每個 SM 獨立運作,同步執行稱為 **warps** 的 32 個執行緒組。為了幫助實現這一點,SM 依賴另一個元件,**warp schedulers:**透過平衡不同 warps 的指令,它們使 SM 能夠透過在一個 warp 被阻塞時切換到另一個 warp 來「隱藏延遲」。這種 **SIMT**(單指令,多執行緒)執行模型意味著 warp 中的所有執行緒在不同資料上同時執行相同的指令。

Warps 的命名參考自紡織,根據 [Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf),這是「第一種並行執行緒技術」。其他 GPU 程式設計模型中 warps 的等效概念包括 WebGPU 中的 [subgroups](https://github.com/gpuweb/gpuweb/pull/4368)、DirectX 中的 [waves](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html),以及 Metal 中的 [simdgroups](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups#2928931)。

SMControlRegistersCoreCoreCoreCoreCoreCoreCoreCoreConstant CacheSharedMemoryL1CacheSMControlRegistersCoreCoreCoreCoreCoreCoreCoreCoreConstant CacheSharedMemoryL1Cache...SMControlRegistersCoreCoreCoreCoreCoreCoreCoreCoreConstant CacheSharedMemoryL1Cache

單個 GPU 內的多個 SMs - [來源](https://www.youtube.com/watch?v=ZQKMZIP3Fzg)

由於數百個 SM 各自並行執行多個 warps,單個 GPU 可以同時執行數萬個執行緒。這種大規模並行性使 GPU 能夠在主導深度學習工作負載的矩陣運算中表現出色!

**精度在討論 FLOPs 時非常重要**。Tensor Cores 可以在不同精度下運作(FP64、FP32、FP16/BF16、FP8、FP4 - 參見[這裡關於浮點數的提醒](https://en.wikipedia.org/wiki/Floating-point_arithmetic))。因此,可實現的輸送量會根據資料類型而劇烈變化,通常相差幾個數量級。較低精度格式能夠實現更高的輸送量,因為它們需要較少的資料移動,並且可以在相同的矽面積中封裝更多運算,但以前由於訓練不穩定性而被避免使用。然而,如今,由於一系列新技術,訓練和推理都越來越推向更低精度,達到 FP8 和 FP4。

如果你想了解更多關於我們使用 FP8 混合精度訓練的經驗,請查看 [Ultra Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。

下表顯示了不同 NVIDIA GPU 世代和精度的理論峰值效能:

Precision\GPU Type | A100 | H100 | H200 | B100 | B200
---|---|---|---|---|---
**FP64** | 9.7 | 34 | 34 | 40 | 40
**FP32** | 19.5 | 67 | 67 | 80 | 80
**FP16/BF16** | 312 | 990 | 990 | 1750 | 2250
**FP8** | - | 3960 | 3960 | 4500 | 5000
**FP4** | - | - | - | 9000 | 10000

_表格顯示根據精度和 GPU 世代的理論 TFLOPs。來源:Nvidia、SemiAnalysis_

較低精度下輸送量的顯著增加不僅僅是關於原始速度,它反映了我們思考數值計算方式的根本轉變。FP8 和 FP4 使模型能夠每**瓦特**和每**秒**執行更多運算,使它們對於大規模訓練和推理都至關重要。H100 在 FP8 下的 3960 TFLOPs 代表比 FP16/BF16 提升 4 倍,而 B200 在 FP4 下的 10,000 TFLOPs 進一步推動了這一點。

**理解這些數字**:這些理論峰值 FLOPs 代表_在理想條件下可實現的最大計算輸送量_,當所有計算單元都被充分利用且資料隨時可用時。在實務上,實際效能在很大程度上取決於你的工作負載能夠多好地為計算單元提供資料,以及你的運算是否能夠有效地映射到可用的硬體上。

對於 SmolLM3,我們將在 NVIDIA H100 80GB HBM3 GPU 上進行訓練,因此我們首先想測試 H100 的理論 TFLOPs 規格與實際效能的對比。為此,我們使用了 [SemiAnalysis GEMM benchmark](https://www.ray.so/#theme=prisma&darkMode=false&code=IyBBTUQgVklQIGltYWdlCmFsaWFzIGRydW49InN1ZG8gZG9ja2VyIHJ1biAtLXByaXZpbGVnZWQgLS1uZXR3b3JrPWhvc3QgLS1kZXZpY2U9L2Rldi9rZmQgLS1kZXZpY2U9L2Rldi9kcmkgLS1ncm91cC1hZGQgdmlkZW8gLS1jYXAtYWRkPVNZU19QVFJBQ0UgLS1zZWN1cml0eS1vcHQgc2VjY29tcD11bmNvbmZpbmVkIC0taXBjPWhvc3QgLS1zaG0tc2l6ZT0xOTI2IC0tcm0gLWl0IgpkcnVuIHNlbWlhbmFseXNpc3dvcmsvYW1kLW1hdG11bDpsYXRlc3QKRElTQUJMRV9BREROX0hJUF9MVD0wIFBZVE9SQ0hfVFVOQUJMRV9PUF9FTkFCTEVEPTEgcHl0aG9uIG1hdG11bC5weQoKI0FNRCBweXBpIG5pZ2h0bHkKZHJ1biBhbWQtbGF0ZXN0LXB5cGktbmlnaHRseS1tYXRtdWwKUFlUT1JDSF9UVU5BQkxFX09QX0VOQUJMRUQ9MSBweXRob24gbWF0bXVsLnB5CgojIEFNRCBweXBpIHN0YWJsZSBQeVRvcmNoIDIuNS4xCmRydW4gc2VtaWFuYWx5c2lzd29yay9hbWQtbGF0ZXN0LXB5cGktc3RhYmxlLW1hdG11bApQWVRPUkNIX1RVTkFCTEVfT1BfRU5BQkxFRD0xIHB5dGhvbiBtYXRtdWwucHkKCiMgTnZpZGlhIHN0YWJsZSAyNC4wOQphbGlhcyBkcnVuPSJkb2NrZXIgcnVuIC0tcm0gLWl0IC0tZ3B1cyBhbGwgLS1pcGM9aG9zdCAtLW5ldD1ob3N0IC0tc2htLXNpemU9MTkyNiIKZHJ1biBzZW1pYW5hbHlzaXN3b3JrL252aWRpYS1tYXRtdWw6bGF0ZXN0CnB5dGhvbiBtYXRtdWwucHkKCg&language=shell):它[測試 Meta Llama 70B 訓練中實際矩陣乘法形狀的輸送量](https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training#general-matrix-multiply-gemm-performance)。

Shape (M, N, K) | FP64 torch.matmul | FP32 torch.matmul | FP16 torch.matmul | BF16 torch.matmul | FP8 TE.Linear (autocast, bias=False) | FP8 torch._scaled_mm (e5m2/e4m3fn) | FP8 torch._scaled_mm (e4m3)
---|---|---|---|---|---|---|---
(16384, 8192, 1280) | 51.5 TFLOPS | 364.5 TFLOPS | 686.5 TFLOPS | 714.5 TFLOPS | 837.6 TFLOPS | 1226.7 TFLOPS | 1209.7 TFLOPS
(16384, 1024, 8192) | 56.1 TFLOPS | 396.1 TFLOPS | 720.0 TFLOPS | 757.7 TFLOPS | 547.3 TFLOPS | 1366.2 TFLOPS | 1329.7 TFLOPS
(16384, 8192, 7168) | 49.5 TFLOPS | 356.5 TFLOPS | 727.1 TFLOPS | 752.9 TFLOPS | 1120.8 TFLOPS | 1464.6 TFLOPS | 1456.6 TFLOPS
(16384, 3584, 8192) | 51.0 TFLOPS | 373.3 TFLOPS | 732.2 TFLOPS | 733.0 TFLOPS | 952.9 TFLOPS | 1445.7 TFLOPS | 1370.3 TFLOPS
(8192, 8192, 8192) | 51.4 TFLOPS | 372.7 TFLOPS | 724.9 TFLOPS | 729.4 TFLOPS | 1029.1 TFLOPS | 1404.4 TFLOPS | 1397.5 TFLOPS

表格顯示 H100 80GB 在 Llama 70B 訓練工作負載中根據精度和矩陣形狀達到的 TFLOPs

**驗證理論效能**:我們的實驗揭示了理論峰值和可實現效能之間的差距。

對於 **FP64 Tensor Core** 運算,我們達到了 49-56 TFLOPs,代表理論峰值(67 TFLOPs)的 74-84%。對於 **TF32**(TensorFloat-32,PyTorch 在 Tensor Cores 上對 FP32 張量預設使用的格式),我們達到了 356-396 TFLOPs,代表理論峰值(~495 TFLOPs 密集)的 72-80%。雖然這些顯示出色的硬體利用率,但這些精度在現代深度學習訓練中很少使用:FP64 由於其計算成本,而 TF32 因為較低精度如 BF16 和 FP8 提供更好的效能。

[NVIDIA 規格](https://www.nvidia.com/en-us/data-center/h100/)通常列出稀疏效能(TF32 為 989 TFLOPs),這假設 2:4 結構化稀疏模式。我們的基準測試測試的密集運算,達到大約稀疏峰值的一半(~495 TFLOPs)。

對於 **BF16** 運算,我們在不同矩陣形狀上一致達到 714-758 TFLOPs,大約是 H100 理論 990 TFLOPs 峰值的 72-77%。實際上,這對於實際工作負載來說是出色的利用率!

Model FLOPs Utilization (MFU)

雖然 kernel 基準測試衡量原始 TFLOPS,但端到端訓練效率由 **Model FLOPs Utilization (MFU)** 捕捉:有用的模型計算與理論峰值硬體效能的比率。

我們的 BF16 matmul 基準測試顯示我們達到了 H100 理論峰值的 72-77%。這代表了我們設定在 kernel 層級可實現的上限。端到端訓練 MFU 必然會更低,因為更複雜的非 matmul 運算、通訊開銷和其他輔助計算。

**訓練中的最先進 MFU**:Meta 在訓練 Llama 3 405B 時達到了 38-41%,而 DeepSeek-v3 在 GPU 上由於與 MoE 架構相關的更緊密通訊瓶頸達到了~20-30%。對於 SmolLM3,我們達到了~30% MFU,我們稍後會看到。大部分差距來自分散式訓練中的節點間通訊開銷。考慮到我們的 kernel 層級上限為~77%,這些端到端數字代表相對於可實現的 matmul 效能約 50-55% 的效率。推理工作負載可以達到更高的 MFU >70%,更接近原始 matmul 效能,儘管來自生產部署的公開結果很少。

**FP8** 結果更加微妙。讓我們看看我們在 3 種不同矩陣乘法方法/kernels 上的結果。

[kernel](https://modal.com/gpu-glossary/device-software/kernel) 是 CUDA 程式碼的單位。

使用 PyTorch 的 `torch._scaled_mm` kernel 與 e4m3 精度,我們根據矩陣形狀達到了 1,210-1,457 TFLOPs,大約是理論 3,960 TFLOPs 峰值的 31-37%。😮 為什麼?這種較低的利用率百分比(在 FP8 中)實際上並不表示效能不佳;相反,它反映出隨著計算輸送量的增長,這些運算越來越受記憶體限制。[Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/) 處理 FP8 資料的速度比記憶體系統能夠提供的速度更快,使記憶體頻寬成為限制因素。

[Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 的 `TE.Linear` 根據形狀達到了 547-1,121 TFLOPs,而 `torch._scaled_mm` 一致提供更高的輸送量。這突顯了一個重要的教訓:_**kernel 實現非常重要**_,即使在針對相同硬體能力時,API 的選擇也可能影響效能 2-3 倍。

對於 SmolLM3 的訓練,這些實際測量幫助我們設定了實際的輸送量期望。在規劃你自己的訓練執行時,使用這些可實現的數字而不是理論峰值來設定你的期望。

Compute Capability

除了選擇正確的 kernel API,我們還需要確保這些 kernels 是為正確的硬體世代編譯的。Compute Capability (CC) 是 NVIDIA 的版本控制系統,它從 PTX 指令集中抽象物理 GPU 細節。它決定了你的 GPU 支援哪些指令和功能。

**為什麼這很重要**:為特定 compute capability 編譯的 kernels 可能無法在較舊的硬體上執行,如果你的程式碼沒有為目標 GPU 的 CC 編譯,你可能會錯過優化。更糟糕的是,框架可能會悄悄選擇次優的 kernels—我們發現 PyTorch 在我們的 H100 上選擇了 sm_75 kernels(compute capability 7.5,為 Turing GPU 設計),導致神秘的減速。這是 [PyTorch 社群中記錄的類似問題](https://discuss.pytorch.org/t/performance-issue-torch-matmul-selecting-cutlass-sm75-kernel-for-a100/220682/3),框架通常預設使用較舊、更相容的 kernels 而不是最優的。這個看似微小的細節可能決定你從相同硬體獲得 720 TFLOPS 還是 500 TFLOPS 的差異。

使用預編譯庫或自訂 kernels 時,始終驗證它們是為你的硬體的 compute capability 建構的,以確保相容性和最佳效能。例如,sm90_xmma_gemm_…_cublas 表示為 SM 9.0(compute capability 9.0,H100 使用)編譯的 kernel。

你可以使用 `nvidia-smi —query-gpu=compute_cap` 檢查你的 GPU 的 compute capability,或在 [NVIDIA CUDA C Programming Guide 的 Compute Capability 部分](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)中找到技術規格。

如我們所見,當計算在低精度下變得太快時,GPU 記憶體似乎成為瓶頸。讓我們看看 GPU 記憶體如何工作,以及是什麼導致瓶頸發生!

#### [GPU 記憶體層次結構:從暫存器到 HBM](https://huggingfacetb-smol-training-playbook.hf.space/#gpu-memory-hierarchy-from-registers-to-hbm)

為了進行計算,GPU 需要讀寫記憶體,因此了解這些傳輸發生的速度很重要。理解 GPU 記憶體層次結構對於編寫高效能 kernels 至關重要。

**TL;DR:** GPU 將記憶體組織成從快但小(暫存器、共享記憶體)到慢但大(HBM 主記憶體)的層次結構。理解這種層次結構至關重要,因為現代 AI 通常受記憶體限制:瓶頸在於移動資料,而不是對其進行計算。operator fusion(如 Flash Attention)透過將中間結果保存在快速晶片上記憶體中而不是寫入慢速 HBM,達到 2-4× 的加速。基準測試顯示 H100 的 HBM3 在實務上提供~3 TB/s,與大型傳輸的理論規格相符。

為了視覺化記憶體操作如何在實務中流經 GPU,讓我們首先看看 [NVIDIA Nsight Compute 的 Memory Chart](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-chart),這是一個分析圖,顯示你選擇的任何 kernel 的資料如何在不同記憶體單元之間移動的圖形表示:

![Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/image_2881384e-bcac-80d6-84fe-d705cb1eae0a.DYXAJOyz_1iK6k0.webp)

Memory Chart 顯示 H100 上 FP64 矩陣乘法期間透過 GPU 記憶體層次結構的資料流

一般來說,Memory Chart 顯示**邏輯單元**(綠色),如 Global、Local、Texture、Surface 和 Shared memory,以及**物理單元**(藍色),如 L1/TEX Cache、Shared Memory、L2 Cache 和 Device Memory。單元之間的連結代表單元之間發生的指令數(Inst)或請求數(Req),顏色表示峰值利用率的百分比:從未使用(0%)到以峰值效能運作(100%)。

你可以使用 [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 為任何 kernel 生成此 memory chart:

```

## Profile a specific kernel with memory workload analysis

ncu --set full --kernel-name "your_kernel_name" --launch-skip 0 --launch-count 1 python your_script.py

## Once profiling is complete, open the results in the Nsight Compute GUI to view the Memory Chart

```

它提供了幾個關鍵見解:
  - **瓶頸識別**:飽和的連結(以紅色/橙色顯示)指示資料移動受限的位置
  - **快取效率**:L1/TEX 和 L2 快取的命中率揭示你的 kernel 如何利用記憶體層次結構
  - **記憶體存取模式**:邏輯單元和物理單元之間的流動顯示你的 kernel 是否具有良好的空間/時間局部性
  - **埠利用率**:即使聚合頻寬看起來未充分利用,個別記憶體埠也可能飽和

在我們上面的具體案例中,你可以看到 kernel 指令如何流經記憶體層次結構(對於我們硬體上的 FP64 矩陣乘法):global load 指令生成對 L1/TEX cache 的請求,這可能命中或未命中並生成對 L2 的進一步請求,最終在未命中時存取 device memory (HBM)。單元內的彩色矩形顯示埠利用率,即使個別連結運作低於峰值,共享資料埠也可能飽和。

優化記憶體層次結構存取

為了達到最佳效能,旨在最小化到較慢記憶體層(HBM)的流量,同時最大化較快層(shared memory、registers)的利用率。

現在讓我們理解使這個圖表成為可能的底層記憶體層次結構。現代 GPU 將記憶體組織成平衡速度、容量和成本的層次結構,這種設計由基本物理和電路約束決定。

![Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/image_2881384e-bcac-801d-9f3d-c875181b9dd1.CtKK2FLa_Z2eUI4H.webp)

H100 (SXM5) GPU 的記憶體層次結構。[來源](https://www.aleksagordic.com/blog/matmul)

在這個層次結構的底部是 **HBM (High Bandwidth Memory):**GPU 的主記憶體,也稱為 global memory 或 device memory。H100 配備了理論頻寬為 3.35 TB/s 的 HBM3。HBM 是記憶體層次結構中最大但最慢的層級。

向上移動層次結構朝向計算單元,我們發現漸進更快但更小的記憶體層級:
  - **L2 cache**:跨 GPU 共享的大型 SRAM 基快取,通常為幾十 MB。在 H100 上,這是 50 MB,頻寬為~13 TB/s
  - **L1 cache 和 Shared Memory (SMEM)**:每個 Streaming Multiprocessor (SM) 都有自己的 L1 cache 和程式設計師管理的 shared memory,它們共享相同的物理 SRAM 儲存。在 H100 上,這個組合空間為每個 SM 256 KB,頻寬為每個 SM ~31 TB/s
  - **Register File (RMEM)**:在層次結構的頂部,暫存器是最快的儲存,直接位於計算單元旁邊。暫存器是個別執行緒私有的,提供以每個 SM ~100s TB/s 測量的頻寬

這種層次結構存在是因為 SRAM(用於快取和暫存器)快速但物理上大且昂貴,而 DRAM(用於 HBM)密集且便宜但較慢。結果:快速記憶體以小數量靠近計算,由漸進更大的較慢記憶體池支援在更遠的地方。

**為什麼這很重要**:理解這種層次結構對於 kernel 優化至關重要。關鍵見解是受記憶體限制的運算受到你可以多快移動資料的限制,而不是你可以多快計算。正如 [Horace He](https://upload.wikimedia.org/wikipedia/commons/b/b2/Hausziege_04.jpg) 在 [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) 中解釋的,_「從記憶體載入」→「自己乘以兩次」→「寫入記憶體」_本質上與_「從記憶體載入」→「自己乘以一次」→「寫入記憶體」_花費相同的時間:與記憶體存取相比,計算是「免費的」。

這就是為什麼 **operator fusion** 如此強大:透過將多個運算組合到單個 kernel 中,你可以將中間結果保存在快速 SRAM 中,而不是在運算之間將它們寫回慢速 HBM。Flash Attention 是這一原則在行動中的完美例子。

Flash Attention:記憶體層次結構優化案例研究

標準 attention 實現受記憶體限制,因為它們在 HBM 中實體化完整的 attention 矩陣:
  1. 計算 `Q @ K^T` → 將 N×N attention 分數寫入 HBM
  2. 應用 softmax → 從 HBM 讀取,計算,寫回 HBM
  3. 乘以 V → 再次從 HBM 讀取 attention 分數

Flash Attention 透過**融合這些運算**並將中間結果保存在 SRAM 中實現其 2-4× 加速:
  - 它不是計算完整的 attention 矩陣,而是處理適合 SRAM 的 attention tiles
  - 中間 attention 分數永遠不會離開快速晶片上記憶體
  - 只有最終輸出被寫回 HBM

**結果**:Flash Attention 將 HBM 存取從 O(N²) 減少到 O(N),將受記憶體限制的運算轉變為更好地利用 GPU 計算能力的運算。這是高效 kernel 設計的本質:_最小化慢速記憶體移動,最大化快速計算_。

**範例:在實務中驗證我們的 HBM3 頻寬**

現在我們理解了記憶體層次結構,讓我們將理論付諸實踐,驗證我們的 H100 GPU 上的實際頻寬!這就是基準測試工具變得至關重要的地方。

**NVBandwidth** 是 NVIDIA 的開源基準測試工具,專門設計用於測量 GPU 系統中的頻寬和延遲。它評估各種記憶體複製模式的資料傳輸速率—host-to-device、device-to-host 和 device-to-device 運算—使用複製引擎和基於 kernel 的方法。該工具對於評估 GPU 間通訊(例如 [NVLink](https://en.wikipedia.org/wiki/NVLink) 和 [PCIe](https://fr.wikipedia.org/wiki/PCI_Express),兩種類型的連接器)和驗證多 GPU 環境中的系統效能特別有價值。

你可以從 [NVIDIA 的 GitHub 儲存庫](https://github.com/NVIDIA/nvbandwidth)安裝 NVBandwidth。該工具輸出詳細的頻寬矩陣,顯示不同裝置之間資料傳輸的效率,使其成為診斷效能瓶頸或驗證健康的 GPU 互連的理想選擇。

讓我們使用它來測量我們的 H100 的本地記憶體頻寬,使用 `device_local_copy` 測試,它測量跨不同訊息大小在 GPU 本地的裝置緩衝區之間的 `cuMemcpyAsync` 頻寬。

cuMemcpyAsync 是一個 CUDA driver API 函數,它非同步複製兩個記憶體指標之間的資料,推斷傳輸類型(host-to-host、host-to-device、device-to-device 或 device-to-host)

```

$ ./nvbandwidth -t device_local_copy -b 2048
memcpy local GPU(column) bandwidth (GB/s)
           0         1         2         3         4         5         6         7
 0   1519.07   1518.93   1519.07   1519.60   1519.13   1518.86   1519.13   1519.33

```

測量的 H100 本地記憶體頻寬
Reset View
Legend

結果揭示了記憶體系統的一個重要特性:**對於小訊息大小(< 1 MB),我們受延遲限制**而不是頻寬限制。啟動記憶體傳輸的開銷主導效能,阻止我們達到峰值頻寬。然而,**對於大訊息大小(≥ 1 MB),我們實現了讀取和寫入運算的持續頻寬~1,500 GB/s**。

由於 HBM 頻寬同時考慮讀取和寫入,我們將這些加起來得到 **3 TB/s 總雙向頻寬**(1,519 讀取 + 1,519 寫入),這接近驗證了 H100 的理論 3.35 TB/s HBM3 規格。

#### [Roofline Model](https://huggingfacetb-smol-training-playbook.hf.space/#roofline-model)

理解你的 kernel 是受計算限制還是受記憶體限制決定了哪些優化會有幫助。

有兩種情況:
  - 如果你**受記憶體限制**(大部分時間花在移動資料上),增加計算輸送量不會有幫助:你需要透過 operator fusion 等技術減少記憶體流量。
  - 如果你**受計算限制**(大部分時間花在 FLOPs 上),優化記憶體存取模式不會有幫助:你需要更多計算能力或更好的演算法。

_roofline model_ 提供了一個視覺框架來理解這些效能特性並識別優化機會。

讓我們將其應用於真實的 kernel 分析。它在我們之前提到的 NSight Compute 分析工具中可用(在「roofline analysis view」下)。這是我們得到的:

![Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/image_2881384e-bcac-80ed-9bdf-c077977d77b8.Dy-eh-v2_Z252kaI.webp)

Roofline 圖顯示 kernel 效能邊界 - 來源:[NVIDIA NSight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

讓我們看看如何閱讀這個圖表,它有兩個軸:
  - **垂直軸(FLOP/s)**:顯示達到的每秒浮點運算,使用對數刻度以容納大範圍的值
  - **水平軸(Arithmetic Intensity)**:代表工作(FLOPs)與記憶體流量(bytes)的比率,以每位元組的 FLOPs 測量。這也使用對數刻度。

roofline 本身由兩個邊界組成:
  - **記憶體頻寬邊界**(斜線):由 GPU 的記憶體傳輸速率(HBM 頻寬)決定。沿著這條線的效能受到資料移動速度的限制。
  - **峰值效能邊界**(平線):由 GPU 的最大計算輸送量決定。沿著這條線的效能受到計算執行速度的限制。

這些邊界相交的 **ridge point** 代表記憶體限制和計算限制機制之間的轉換。

我們可以透過查看圖表的兩個分區來解釋效能:
  - **記憶體限制**(斜線邊界下方):此區域中的 Kernels 受記憶體頻寬限制。GPU 正在等待資料,增加計算能力不會有幫助。優化應該專注於透過 operator fusion、更好的記憶體存取模式或增加算術強度等技術減少記憶體流量。
  - **計算限制**(平線邊界下方):此區域中的 Kernels 受計算輸送量限制。GPU 有足夠的資料但無法足夠快地處理它。優化應該專注於演算法改進或利用 Tensor Cores 等專用硬體。

**達到的值**(繪製的點)顯示你的 kernel 目前所處的位置。從這個點到 roofline 邊界的距離代表你的優化空間,越接近邊界,你的 kernel 效能越優。

在我們的例子中,kernel 位於記憶體限制區域,表明透過優化記憶體流量仍有改進空間!

要更深入了解 GPU 內部,包括 CUDA cores、Tensor Cores、記憶體層次結構和低階優化技術的詳細解釋,請查看 [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)!現在我們理解了 GPU _內部_發生的事情,讓我們放大視角,探索 GPU 如何與世界其他部分通訊。
