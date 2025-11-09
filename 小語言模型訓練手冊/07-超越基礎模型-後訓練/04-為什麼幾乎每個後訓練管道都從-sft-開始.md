### [為什麼(幾乎)每個後訓練管道都從 SFT 開始](https://huggingfacetb-smol-training-playbook.hf.space/#why-almost-every-post-training-pipeline-starts-with-sft)

如果你最近在 X 上花時間,你會認為強化學習(RL)是唯一的遊戲。每天都會帶來新的縮寫、[演算法調整](https://x.com/agarwl_/status/1981518825007853891)以及關於 RL 是否可以引發新能力的激烈辯論 ([Chu et al., 2025](https://arxiv.org/abs/2501.17161); [Yue et al., 2025](https://arxiv.org/abs/2504.13837))。
正如我們稍後在本章中將看到的,RL 確實有效,但會帶來我們在下面討論的實際權衡。
當然,RL 並不新鮮。OpenAI 和其他實驗室在很大程度上依賴人類回饋的 RL(RLHF)([Lambert et al., 2022](https://huggingfacetb-smol-training-playbook.hf.space/#bib-rlhf))來對齊他們的早期模型,但直到 DeepSeek-R1 ([DeepSeek-AI, Guo, et al., 2025](https://arxiv.org/abs/2501.12948)) 的發布,基於 RL 的後訓練才真正在開源生態系統中流行起來。
但有一件事沒有改變:幾乎每個有效的後訓練管道仍然從監督式微調(SFT)開始。原因很簡單:
  - **它很便宜:** 與 RL 相比,SFT 需要適度的計算。你通常可以獲得有意義的收益,而無需燃燒大量的矽,並且所需時間只是 RL 的一小部分。
  - **它很穩定:** 與 RL 不同,RL 以對獎勵設計和超參數敏感而聞名,SFT「就是有效」。
  - **它是正確的基線:** 一個好的 SFT 檢查點通常會給你大部分你追求的收益,並且它使 DPO 或 RLHF 等後續方法更加有效。
在實踐中,這意味著 SFT 不僅僅是第一步,因為它很容易;它是在嘗試任何更複雜的事情之前始終如一地改善效能的步驟。當你使用基礎模型時,這一點尤其正確。除了少數例外,基礎模型太粗糙,無法從高級後訓練方法中受益。
DeepSeek R1-Zero 怎麼樣?
在前沿,從 SFT 開始的通常原因並不總是適用。沒有更強的模型可以蒸餾,人類註釋對於像長思維鏈這樣的複雜行為來說太嘈雜。這就是為什麼 DeepSeek 跳過 SFT 並直接使用 R1-Zero 進行 RL;以_發現_無法用標準監督教導的推理行為。
如果你處於那種狀態,從 RL 開始可能是有道理的。但如果你在那裡操作...你可能無論如何都不會閱讀這篇部落格文章 😀。
因此,如果 SFT 是大多數管道開始的地方,下一個問題是:你應該微調_什麼_?這從選擇正確的基礎模型開始。

#### [選擇基礎模型](https://huggingfacetb-smol-training-playbook.hf.space/#picking-a-base-model)

在選擇用於後訓練的基礎模型時,一些實際維度最重要:
  - **模型大小:** 儘管小型模型隨著時間的推移已經大幅改善,但今天仍然是較大模型泛化得更好,而且通常需要更少的樣本。選擇一個代表你計劃在訓練後如何使用或部署模型的模型大小。在 [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min%3A0%2Cmax%3A32B&sort=trending) 上,你可以按模態和大小過濾模型以找到合適的候選者:
![Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/Screenshot_2025-10-24_at_09_37_24_2961384e-bcac-8055-9e8e-ffbd3a1aa368.C0macOQ5_Z1iRGEy.webp)
  - **架構(MoE vs dense):** MoE 模型每個 token 啟動參數的子集,並為每單位計算提供更高的容量。它們非常適合大規模服務,但根據我們的經驗,微調起來更棘手。相比之下,密集模型更容易訓練,並且在較小規模上通常優於 MoE。
  - **後訓練記錄:** 基準測試很有用,但如果基礎模型已經產生了一系列與社群產生共鳴的強大後訓練模型,那就更好了。這提供了模型訓練是否良好的代理。
[LocalLLaMa subreddit](https://www.reddit.com/r/LocalLLaMA/) 是了解新模型廣泛氛圍的好地方。[Artificial Analysis](https://artificialanalysis.ai) 和 [LMArena](https://lmarena.ai) 也提供新模型的獨立評估,儘管這些平台有時會被[模型提供者進行基準測試最佳化。](https://huggingface.co/papers/2504.20879)
根據我們的經驗,來自 Qwen、Mistral 和 DeepSeek 的基礎模型最適合後訓練,Qwen 是明顯的最愛,因為每個模型系列通常涵蓋大範圍的參數(例如,Qwen3 模型的大小範圍從 0.6B 到 235B!)。這個特性使擴展變得更加簡單。
一旦你選擇了符合你部署需求的基礎模型,下一步就是建立一個簡單、快速的 SFT 基線來探測其核心技能。

#### [訓練簡單基線](https://huggingfacetb-smol-training-playbook.hf.space/#training-simple-baselines)

對於 SFT,一個好的基線應該快速訓練,專注於模型的核心技能,並且在特定能力不夠時容易擴展更多資料。選擇哪些資料集用於初始基線涉及一些品味和對那些可能具有高品質的資料集的熟悉度。一般來說,避免過度關注在學術基準測試上報告高分數的公共資料集,而是專注於那些已用於訓練優秀模型的資料集,如 [OpenHermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5)。例如,在 SmolLM1 的開發中,我們最初在 [WebInstruct](https://huggingface.co/datasets/TIGER-Lab/WebInstructFull) 上執行 SFT,這在紙面上是一個很棒的資料集。然而,在我們的氛圍測試期間,我們發現它太專注於科學,因為模型會用方程式回應簡單的問候語,如「你好嗎?」。
使用氛圍測試來發現訓練資料中的怪癖是本章的一個反覆出現的主題 — 不要低估與你的模型聊天的力量!
這導致我們創建了 [Everyday Conversations](https://huggingfacetb-smol-training-playbook.hf.space/2421384ebcac800cb22cdf0bb34c69f7) 資料集,事實證明這對於在小型模型中灌輸基本聊天能力至關重要。
對於 SmolLM3,我們著手訓練一個混合推理模型,最初選擇了一小組資料集來針對推理、指令跟隨和可控性。下表顯示了每個資料集的統計資訊:
資料集 | 推理模式 | # 範例 | % 範例 | # tokens (M) | % tokens | 每個範例平均 # tokens | 上下文中平均 # tokens | 回應中平均 # tokens | 平均 # 輪數

---|---|---|---|---|---|---|---|---|---
Everyday Conversations | /no_think | 2,260 | 2.3 | 0.6 | 0.8 | 260.2 | 222.3 | 94.0 | 7.8
SystemChats 30k | /no_think | 33,997 | 35.2 | 21.5 | 28.2 | 631.9 | 422.8 | 267.7 | 6.3
Tulu 3 SFT Personas IF | /no_think | 29,970 | 31.0 | 13.3 | 17.5 | 444.5 | 119.8 | 380.7 | 2
Everyday Conversations (Qwen3-32B) | /think | 2,057 | 2.1 | 3.1 | 4.1 | 1,522.4 | 376.8 | 1,385.6 | 4
SystemChats 30k (Qwen3-32B) | /think | 27,436 | 28.4 | 29.4 | 38.6 | 1070.8 | 84.6 | 1,042.7 | 2
s1k-1.1 | /think | 835 | 0.9 | 8.2 | 10.8 | 8,859.3 | 370.9 | 9,728.5 | 2
Total | - | 96,555 | 100.0 | 76.1 | 100.0 | 2,131.5 | 266.2 | 2,149.9 | 4.0

混合推理基線的資料混合
正如我們在 SmolLM3 的開發過程中了解到的,訓練混合推理模型比標準 SFT 更棘手,因為你不能只是將資料集混合在一起;你需要跨模式_配對_資料。每個範例都必須清楚地指示模型應該進行擴展推理還是給出簡潔答案,理想情況下你希望有平行範例來教它何時切換模式。從上表中要注意的另一件事是,你應該根據_tokens_而不是_範例_來平衡你的資料混合:例如,s1k-1.1 資料集約佔總範例的 1%,但由於長推理回應,約佔總 tokens 的 11%。
這給了我們對我們最關心的技能的基本覆蓋,但也引入了一個新挑戰:每個資料集必須根據是否應該啟用擴展思考而進行不同的格式化。為了統一這些格式,我們需要一個一致的聊天模板。

#### [選擇好的聊天模板](https://huggingfacetb-smol-training-playbook.hf.space/#picking-a-good-chat-template)

當涉及到選擇或設計聊天模板時,沒有一刀切的答案。在實踐中,我們發現預先詢問幾個問題是值得的:
  - **使用者可以自訂系統角色嗎?** 如果使用者應該能夠定義自己的系統提示(例如「像海盜一樣行事」),模板需要清楚地處理它。
  - **模型需要工具嗎?** 如果你的模型需要呼叫 API,模板需要適應工具呼叫和回應的結構化輸出。
  - **它是推理模型嗎?** 推理模型使用像 `<think> ... </think>` 這樣的模板來將模型的「思想」與其最終答案分開。一些模型在對話的輪次中丟棄推理 tokens,聊天模板需要處理該邏輯。
  - **它是否與推理引擎一起工作?** 像 vLLM 和 SGLang 這樣的推理引擎有專用的推理和工具解析器。與這些解析器的相容性後來節省了很多痛苦,特別是在複雜的代理基準測試中,一致的工具呼叫至關重要。
下表顯示了一些流行的聊天模板以及它們在關鍵考慮因素上的比較:
**聊天模板** | **系統角色自訂** | **工具** | **推理** | **推理相容性** | **註釋**
---|---|---|---|---|---
ChatML | ✅ | ✅ | ❌ | ✅ | 簡單且適用於大多數用例。
Qwen3 | ✅ | ✅ | _✅_ | ✅ | 混合推理模板
DeepSeek-R1 | ❌ | ❌ | ✅ | ✅ | 使用 `<think>` 預填充推理內容。
Llama 3 | ✅ | ✅ | ❌ | ✅ | 有內建工具,如 Python 程式碼解釋器。
Gemma 3 | ✅ | ❌ | ❌ | ❌ | 系統角色自訂在第一個使用者輪次定義。
Command A Reasoning | ✅ | ✅ | ✅ | ❌ | 每個模型有多個聊天模板。
GPT-OSS | ✅ | ✅ | ✅ | ✅ | 基於 [Harmony 回應格式](https://cookbook.openai.com/articles/openai-harmony)。複雜但多功能。

在大多數情況下,我們發現 ChatML 或 Qwen 的聊天模板是一個很好的起點。對於 SmolLM3,我們需要一個混合推理的模板,發現 Qwen3 是為數不多的在我們關心的維度上取得良好平衡的模板之一。然而,它有一個我們不完全滿意的怪癖:推理內容在對話中除最後一輪外的所有輪次中都被_丟棄_。如下圖所示,這類似於 [OpenAI 的推理模型如何工作](https://platform.openai.com/docs/guides/reasoning/how-reasoning-works):

```

flowchart LR
    subgraph Turn1 ["Turn 1"]
        T1_Input["**INPUT**"]
        T1_Output["**OUTPUT**"]
    end

    subgraph Turn2 ["Turn 2"]
        T2_Input["**INPUT**"]
        T2_Output["**OUTPUT**"]
    end

    subgraph Turn3 ["Turn 3"]
        T3_Input["**INPUT**"]
        T3_Output1["**OUTPUT**"]
        Reasoning["**REASONING**"]
        T3_Output2_Top["**OUTPUT**"]
        TruncatedOutput["**TRUNCATED OUTPUT**"]
    end

    T1_Input --> T2_Input
    T1_Output --> T2_Input

    T2_Input --> T3_Input
    T2_Output --> T3_Input

    T3_Input --> T3_Output1
    T3_Output1 --> Reasoning
    Reasoning --> T3_Output2_Top

    T3_Output2_Top -->|CONTEXT WINDOW ✂️| TruncatedOutput

    classDef input fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    classDef output fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef reasoning fill:#f5f5f5,stroke:#999,stroke-width:1px
    classDef truncated fill:#ffebee,stroke:#f44336,stroke-width:3px
    classDef subgraphStyle fill:#f8f9fa,stroke:#dee2e6,stroke-width:1px
    classDef linkLabel fill:#f8f9fa,stroke:#dee2e6,stroke-width:1px

    class T1_Input,T2_Input,T3_Input input
    class T1_Output,T2_Output,T3_Output1,T3_Output2_Top output
    class Reasoning reasoning
    class TruncatedOutput truncated
    class Turn1,Turn2,Turn3 subgraphStyle
    linkStyle 4 stroke:#333,stroke-width:2px,fill:#f8f9fa

```

儘管這對_推理_有意義(以避免爆炸上下文),但我們得出結論,對於_訓練_來說,重要的是_在所有輪次中保留推理 tokens_,以便適當地調整模型。
相反,我們決定製作我們自己的聊天模板,具有以下功能:
  - 一個結構化的系統提示,如 [Llama 3 的](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=meta-llama%2FLlama-3.1-8B-Instruct)和那些[從專有模型中越獄](https://github.com/elder-plinius/CL4R1T4S)的那些。我們還希望提供完全覆蓋系統提示的靈活性。
  - 支援 [code agents](https://huggingface.co/learn/agents-course/en/unit2/smolagents/code_agents),它執行任意 Python 程式碼而不是進行 JSON 工具呼叫。
  - 透過系統訊息明確控制推理模式。
為了迭代聊天模板的設計,我們使用了 [Chat Template Playground](https://huggingface.co/spaces/huggingfacejs/chat-template-playground)。這個方便的應用程式是由我們在 Hugging Face 的同事開發的,使預覽訊息如何呈現和除錯格式問題變得容易。以下是嵌入版本的遊樂場,因此你可以直接嘗試它:
從下拉清單中選擇不同的範例,以查看聊天模板如何適用於多輪對話、推理或工具使用。你甚至可以手動更改 JSON 輸入以啟用不同的行為。例如,看看如果你提供 `enable_thinking: false` 或將 `/no_think` 附加到系統訊息會發生什麼。
一旦你確定了一些初始資料集和聊天模板,就該訓練一些基線了!

#### [嬰兒基線](https://huggingfacetb-smol-training-playbook.hf.space/#baby-baselines)

在我們深入優化和擠壓每一點效能之前,我們需要建立一些「嬰兒基線」。這些基線不是要達到最先進的水平(還沒有),而是旨在驗證聊天模板是否按你想要的方式工作,以及初始超參數集是否產生穩定的訓練。只有在我們有了這個基礎之後,我們才開始大量調整超參數和訓練混合。
當涉及到訓練 SFT 基線時,以下是要考慮的主要事項:
  - 你將使用完全微調(FullFT)還是像 LoRA 或 QLoRA 這樣的參數高效方法?正如 Thinking Machines 的精彩[部落格文章](https://thinkingmachines.ai/blog/lora/)中所描述的,LoRA 可以在某些條件下匹配 FullFT(通常由資料集的大小決定)。
  - 你需要什麼類型的並行性?對於小型模型或使用 LoRA 訓練的模型,你通常可以使用資料並行。對於較大的模型,你將需要 FSDP2 或 DeepSpeed ZeRO-3 來共享模型權重和優化器狀態。對於使用長上下文訓練的模型,請使用像 [context parallelism](https://huggingface.co/docs/trl/v0.23.0/en/reducing_memory_usage#context-parallelism) 這樣的方法。
  - 如果你的硬體支援,請使用 FlashAttention 和 Liger 等內核。許多這些內核託管在 [Hugging Face Hub](https://huggingface.co/models?other=kernel) 上,可以透過 TRL 中的[簡單參數](https://huggingface.co/docs/trl/kernels_hub)設定,以顯著降低 VRAM 使用量。
  - 遮罩損失以[僅在助理 tokens 上訓練](https://huggingface.co/docs/trl/sft_trainer#train-on-assistant-messages-only)。正如我們在下面討論的,這可以透過使用特殊的 `{% generation %}` 關鍵字包裝聊天模板的助理輪次來實現。
  - 調整學習率;除了資料之外,這是決定你的模型是「還行」還是「很棒」的最重要因素。
  - [打包訓練樣本](https://huggingface.co/docs/trl/v0.23.0/en/reducing_memory_usage#packing)並調整序列長度以匹配你的資料分布。這將顯著加快訓練速度。TRL 有一個方便的[應用程式](https://huggingface.co/docs/trl/v0.23.0/en/reducing_memory_usage#how-to-choose-the-maxlength-value)為你執行此操作。
讓我們看看這些選擇如何為 SmolLM3 產生結果。對於我們的第一個基線實驗,我們想要一個簡單的健全性檢查:聊天模板是否真的引發混合推理?為了測試這一點,我們從我們的[表](https://huggingfacetb-smol-training-playbook.hf.space/#sft-datasets)中比較了三個資料混合:
  - **Instruct:** 在非推理範例上訓練。
  - **Thinking:** 在推理範例上訓練。
  - **Hybrid:** 在所有範例上訓練。
對於每個混合,我們使用 FullFT 在 [SmolLM3-3B-Base](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base) 上執行 SFT,學習率為 1e-5,有效批次大小為 128,訓練 1 個 epoch。
我們發現對於大多數模型和資料集,這種超參數選擇作為基線效果很好。
由於這些是小型資料集,我們沒有使用打包,將序列限制在 Instruct 子集的 8,192 tokens 和其餘的 32,768 tokens。在一個 8 x H100 的節點上,這些實驗執行得很快,根據子集需要 30-90 分鐘。下圖比較了每個子集對應推理模式的效能:
這些結果很快向我們展示了混合模型表現出一種「分裂大腦」,其中一種推理模式的資料混合對另一種影響很小。這在大多數評估中明顯,Instruct、Thinking 和 Hybrid 子集之間具有相似的分數,LiveCodeBench v4 和 IFEval 是例外,混合資料提高了整體效能。

#### [**氛圍測試你的基線**](https://huggingfacetb-smol-training-playbook.hf.space/#vibe-test-your-baselines)

儘管評估看起來不錯,但當我們嘗試讓混合模型以不同的角色行事(例如像海盜)時,它始終忽略我們在系統訊息中放置的任何內容。經過一番挖掘後,我們發現原因是由於我們格式化資料的方式:
![Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/Screenshot_2025-09-26_at_22_36_40_27a1384e-bcac-8063-94e0-f1c689e7d9b9.vtcw08KN_wObJG.webp)
發生的情況是,在我們的聊天模板設計中,我們暴露了一個 `custom_instructions` 參數來儲存系統提示。例如,以下是我們如何在對話中設定角色:

```

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
messages = [
    {
        "content": "I'm trying to set up my iPhone, can you help?",
        "role": "user",
    },
    {
        "content": "Of course, even as a vampire, technology can be a bit of a challenge sometimes [TRUNCATED]",
        "role": "assistant",
    },
]
chat_template_kwargs = {
    "custom_instructions": "You are a vampire technologist",
    "enable_thinking": False,
}
rendered_input = tok.apply_chat_template(messages, tokenize=False, **chat_template_kwargs)
print(rendered_input)

## <|im_start|>system

### Metadata

## Knowledge Cutoff Date: June 2025

## Today Date: 28 October 2025

## Reasoning Mode: /no_think

### Custom Instructions

## You are a vampire technologist

## <|im_start|>user

## I'm trying to set up my iPhone, can you help?<|im_end|>

## <|im_start|>assistant

## <think>

## </think>

## Of course, even as a vampire, technology can be a bit of a challenge sometimes # [TRUNCATED]<|im_end|>

```

問題是我們的資料樣本看起來像這樣:

```

{
    "messages": [
        {
            "content": "I'm trying to set up my iPhone, can you help?",
            "role": "user",
        },
        {
            "content": "Of course, even as a vampire, technology can be a bit of a challenge sometimes [TRUNCATED]",
            "role": "assistant",
        },
    ],
    "chat_template_kwargs": {
        "custom_instructions": None,
        "enable_thinking": False,
        "python_tools": None,
        "xml_tools": None,
    },
}

```

我們處理程式碼中的一個錯誤將 `custom_instructions` 設定為 `None`,這有效地從_每個訓練樣本_中移除了系統訊息 🙈!因此,我們沒有為這些訓練樣本獲得一個好的角色,而是最終得到了 SmolLM3 預設系統提示:

```

chat_template_kwargs = {"custom_instructions": None, "enable_thinking": False}
rendered_input = tok.apply_chat_template(messages, tokenize=False, **chat_template_kwargs)
print(rendered_input)

## <|im_start|>system

#### Metadata

## Knowledge Cutoff Date: June 2025

## Today Date: 28 October 2025

## Reasoning Mode: /no_think

#### Custom Instructions

## You are a helpful AI assistant named SmolLM, trained by Hugging Face.

## <|im_start|>user

## I'm trying to set up my iPhone, can you help?<|im_end|>

## <|im_start|>assistant

## <think>

## </think>

## Of course, even as a vampire, technology can be a bit of a challenge sometimes [TRUNCATED]<|im_end|>

```

這對 SystemChats 子集特別有問題,其中所有角色都透過 `custom_instructions` 定義,因此模型傾向於在對話中期隨機切換角色。這將我們帶到以下規則:
規則
始終氛圍測試你的模型,即使評估看起來不錯。更多時候,你會發現訓練資料中的微妙錯誤。
修復這個錯誤對評估沒有影響,但最終我們確信聊天模板和資料集格式化正在工作。一旦你的設置穩定並且你的資料管道檢查出來,下一步就是專注於開發特定能力。

#### [針對特定能力](https://huggingfacetb-smol-training-playbook.hf.space/#targeting-specific-capabilities)

在 [Open-R1](https://github.com/huggingface/open-r1) 的開發過程中,我們注意到在單輪推理資料上完全訓練基礎模型將無法泛化到多輪。這並不奇怪;沒有這樣的範例,模型正在其訓練分布之外進行測試。
為了為 SmolLM3 定量衡量這一點,我們從 Qwen3 中汲取靈感,他們開發了一個名為 _ThinkFollow_ 的內部評估,它隨機插入 `/think` 或 `/no_think` 標籤來測試模型是否可以一致地切換推理模式。在我們的實作中,我們從 Multi-IF 中取得提示,然後檢查模型是否生成了包含在 `<think>` 和 `</think>` 標籤中的空或非空思考塊。正如預期的那樣,來自我們混合基線的結果顯示模型在第一輪之後啟用推理模式的失敗慘重:
為了修復這個能力,我們構建了一個名為 IFThink 的新資料集。基於 Multi-IF 管道,我們使用了來自 [Tulu 3 的指令跟隨子集](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following)的單輪指令,並使用 Qwen3-32B 將它們擴展為多輪交換,以生成可驗證的指令和推理軌跡。該方法如下所示:
我們考慮過過濾掉衝突的指令,但初始結果足夠強大,可以跳過這一步。

```

flowchart TD
    %% Inputs
    IFEval["Tülu3 IF Dataset"]
    InstructionTypes["Set of instruction types"]
       %% English Multi-Turn Generation
    SingleTurn["Single turn prompt"]
    LLM1["Generate instructions with Qwen3-32B"]
       subgraph Turns ["Multi-turn prompts"]
        Turn1["Prompt @ turn 1"]
        Turn2["Prompt @ turn 2"]
        Turn3["Prompt @ turn 3"]
    end
       MultiTurn["Generate reasoning traces with Qwen3-32B"]
       IFThink["IFThink"]
       %% Connections
    IFEval --> SingleTurn
    IFEval --> InstructionTypes
    SingleTurn --> Turn1
    InstructionTypes --> LLM1
    LLM1 --> Turn2
    LLM1 --> Turn3
       Turn1 --> MultiTurn
    Turn2 --> MultiTurn
    Turn3 --> MultiTurn
       MultiTurn --> IFThink
       %% Styling
    classDef question fill:#ffd0c5
    classDef decision fill:#f9f9f9
    classDef success fill:#d1f2eb
    classDef danger fill:#fef3c7
    classDef category fill:#fef3c7
       class IFEval,InstructionTypes question
    class SingleTurn,LLM1,MultiTurn decision
    class Turn1,Turn2,Turn3 decision
    class IFThink success

```

在我們的基線混合中包含這些資料產生了顯著的改善:
在使用 IFThink 修復多輪推理問題後,我們的基線終於按預期運作;它可以在輪次之間保持一致,遵循指令,並正確使用聊天模板。有了這個基礎,我們回到基礎:調整訓練設置本身。

#### [哪些超參數實際上很重要?](https://huggingfacetb-smol-training-playbook.hf.space/#which-hyperparameters-actually-matter)

在 SFT 中,實際上只有少數超參數很重要。學習率、批次大小和打包幾乎決定了你的模型訓練的效率以及它泛化的好壞。在我們的嬰兒基線中,我們選擇了合理的預設值,只是為了驗證資料和聊天模板。現在設置已經穩定,我們重新審視這些選擇,看看它們對我們的基線有多大影響。
**遮罩使用者輪次**
聊天模板的一個微妙設計選擇是在訓練期間是否遮罩使用者輪次。在大多數聊天風格資料集中,每個訓練範例由交替的使用者和助理訊息組成(可能帶有交錯的工具呼叫)。如果我們訓練模型預測所有 tokens,它實際上是學習自動完成使用者查詢,而不是專注於產生高品質的助理回應。
如下圖所示,遮罩使用者輪次透過確保模型的損失僅在助理輸出上計算,而不是使用者訊息上計算,來防止這種情況:
input_ids
user
assistant
user
assistant
labels
在 TRL 中,為可以回傳助理 tokens 遮罩的聊天模板應用遮罩。在實踐中,這涉及在模板中包含一個 `{% generation %}` 關鍵字,如下所示:

```

{%- for message in messages -%}
  {%- if message.role == "user" -%}
    {{ "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n" }}

  {%- elif message.role == "assistant" -%}
{% generation %}
{{ "<|im_start|>assistant" + "\n" + message.content + "<|im_end|>\n" }}

{% endgeneration %}
  {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
  {{ "<|im_start|>assistant\n" }}

{%- endif %}

```

然後,當 `apply_chat_template()` 與 `return_assistant_tokens_mask=True` 一起使用時,聊天模板將指示對話的哪些部分應該被遮罩。這是一個簡單的範例,顯示助理 tokens 被賦予 ID 1,而使用者 tokens 被遮罩為 ID 0:

```

chat_template = '''
{%- for message in messages -%}
  {%- if message.role == "user" -%}
    {{ "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n" }}

  {%- elif message.role == "assistant" %}
    {% generation %}
    {{ "<|im_start|>assistant" + "\n" + message.content + "<|im_end|>\n" }}

    {% endgeneration %}
  {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
  {{ "<|im_start|>assistant\n" }}

{%- endif %}
'''
rendered_input = tok.apply_chat_template(messages, chat_template=chat_template, return_assistant_tokens_mask=True, return_dict=True)
print(rendered_input)

## {'input_ids': [128011, 882, 198, 40, 2846, 4560, 311, 743, 709, 856, 12443, 11, 649, 499, 1520, 30, 128012, 198, 257, 128011, 78191, 198, 2173, 3388, 11, 1524, 439, 264, 51587, 11, 5557, 649, 387, 264, 2766, 315, 264, 8815, 7170, 510, 2434, 12921, 9182, 60, 128012, 271], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'assistant_masks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

```

在實踐中,遮罩對下游評估沒有巨大影響,通常在大多數情況下提供幾個點的改善。對於 SmolLM3,我們發現它對 IFEval 影響最大,可能是因為模型不太傾向於重述提示並更密切地遵循各種約束。下圖顯示了使用者遮罩如何影響每個評估和推理模式的比較:
**打包還是不打包?**
序列打包是對訓練效率產生巨大差異的訓練細節之一。在 SFT 中,大多數資料集包含可變長度的樣本,這意味著每個批次包含大量浪費計算並減慢收斂的填充 tokens。
打包透過將多個序列連接在一起直到達到所需的最大 token 長度來解決這個問題。有各種執行連接的方法,TRL 採用「最佳擬合遞減」策略 ([Ding et al., 2024](https://arxiv.org/abs/2404.10830)),其中要打包的序列順序由它們的長度決定。如下所示,此策略最小化了批次邊界上文件的截斷,同時也減少了填充 tokens 的數量:
Dataset
Packing = False
Packing = True
後訓練 vs 預訓練中的打包
在預訓練中,這實際上不是一個問題。當在數兆 tokens 上訓練時,打包對於避免在填充上浪費大量計算至關重要。像 Megatron-LM 和 Nanotron 這樣的預訓練框架預設實作打包。後訓練是不同的。因為執行較短,權衡會改變。
為了了解打包對訓練的效率如何,下面我們比較了在我們基線資料集的一個 epoch 上打包和無打包之間的執行時間:
在有效批次大小為 32 之後執行時間變平的原因是,這是在不呼叫梯度累積的情況下可能的最大大小。
根據批次大小,我們看到打包將輸送量提高了 3-5 倍!那麼,你_總是_應該使用打包嗎?在某種程度上,答案取決於你的資料集有多大,因為打包透過將更多 tokens 裝入每個步驟來減少每個 epoch 的優化步驟數。你可以在下圖中看到這一點,其中我們繪製了每個批次的非填充 tokens 的平均數量:
使用打包,每個批次的 tokens 數量隨批次大小線性擴展,與不打包訓練相比,每個優化步驟可以包含多達 33 倍的 tokens!然而,打包可能會略微改變訓練動態:雖然你處理的資料總體上更多,但你進行的梯度更新更少,這可能會影響最終效能,特別是在小型資料集上,每個樣本都更重要。例如,如果我們在相同的有效批次大小 128 下比較打包與無打包,我們會看到某些評估(如 IFEval)的效能顯著下降近 10 個百分點:
更一般地說,我們看到一旦有效批次大小大於 32,對於這個特定模型和資料集,效能平均會下降:
在實踐中,對於資料集龐大的大規模 SFT,打包幾乎總是有益的,因為計算節省遠遠超過梯度頻率的任何微小差異。然而,對於較小或更多樣化的資料集 — 如特定領域微調或在有限人工策劃資料上的指令調整 — 可能值得禁用打包以保留樣本粒度,並確保每個範例對優化做出清晰貢獻。
最終,最佳策略是經驗性的:從啟用打包開始,監控輸送量和下游評估,並根據速度增益是否轉化為等效或改善的模型品質進行調整。
**調整學習率**
我們現在來到最後一個但仍然重要的超參數:學習率。設定得太高,訓練可能會發散;太低,收斂會痛苦地緩慢。
在 SFT 中,最佳學習率通常比預訓練期間使用的小一個數量級(或更多)。這是因為我們從具有豐富表示的模型初始化,激進的更新可能導致災難性遺忘。
後訓練 vs 預訓練中的學習率調整
與預訓練不同,在完整執行上進行超參數掃描成本過高,後訓練執行足夠短,我們實際上可以進行完整的學習率掃描。
在我們的實驗中,我們發現「最佳」學習率隨模型系列、大小和打包的使用而變化。由於高學習率可能導致梯度爆炸,我們發現在啟用打包時略微降低學習率通常更安全。你可以在下面看到這一點,使用 3e-6 或 1e-5 的小學習率比大值提供更好的整體效能:
在選擇要掃描的學習率值範圍時,我們發現選擇像 [1e-6, 3e-6, 1e-5, 3e-5, 1e-4] 這樣的初始範圍很有用。這涵蓋了兩個數量級,並允許我們縮小到可以應用一些額外調整的區域
儘管平均幾個點可能看起來不多,但如果你查看像 AIME25 這樣的個別基準測試,當學習率大於 1e-5 時,你會看到效能急劇下降。
**擴展 epochs 數量**
在我們的消融實驗中,我們通常訓練一個 epoch 以快速迭代。一旦你確定了良好的資料混合並調整了學習率等關鍵參數,下一步就是增加最終訓練的 epochs 數量。
例如,如果我們取我們的基線資料混合並訓練五個 epochs,我們會看到可以在平均效能上擠出更多百分點:
正如我們在學習率掃描中看到的,平均效能掩蓋了擴展 epochs 數量對個別評估的影響:在使用擴展思考的 LiveCodeBench v4 的情況下,我們幾乎將一個 epoch 的效能翻倍!
一旦你迭代了你的 SFT 資料混合並且模型達到了合理的效能水平,下一步通常是探索更高級的方法,如 [偏好優化](https://huggingfacetb-smol-training-playbook.hf.space/#from-sft-to-preference-optimisation:-teaching-models-what- _better-_ means) 或 [強化學習](https://huggingfacetb-smol-training-playbook.hf.space/#going-online-and-beyond-supervised-labels)。然而,在深入研究這些之前,值得考慮額外的計算是否更好地花在透過_持續預訓練_來加強基礎模型上。
後訓練中的優化器
我們在預訓練部分提到的另一個重要組件是優化器。同樣,AdamW 仍然是後訓練的預設選擇。一個開放的問題是,使用像 Muon 這樣的替代優化器進行預訓練的模型是否應該使用_相同的_優化器進行後訓練。Kimi 團隊發現,對他們的 [Moonlight](https://arxiv.org/abs/2502.16982) 模型使用相同的優化器進行預訓練和後訓練產生了最佳效能。

#### [透過持續預訓練提升推理](https://huggingfacetb-smol-training-playbook.hf.space/#boosting-reasoning-through-continued-pretraining)

持續預訓練 — 或者如果你想聽起來很花哨的話,中期訓練 — 意味著取一個基礎模型,在進行 SFT 之前在大量特定領域的 tokens 上進一步訓練它。當你的 SFT 目標能力共享一個共同的核心技能(如編碼或推理)時,中期訓練很有用。在實踐中,這將模型轉向更好地支援推理、特定語言或你關心的任何其他能力的分布。從已經整合該核心技能的模型開始 SFT 允許你的模型更好地專注於你的 SFT 資料中的特定主題,而不是使用計算從頭學習核心技能。
中期訓練方法可以追溯到 ULMFit ([Howard & Ruder, 2018](https://arxiv.org/abs/1801.06146)),它開創了通用預訓練 → 中期訓練 → 後訓練的三階段管道,現在在像 FAIR 的 Code World Model ([team et al., 2025](https://arxiv.org/abs/2510.02387)) 這樣的現代 LLM 中很常見:
Pre-Training
General
Pre-training
8T tokens
8k context
Code World Modeling
Mid-training
5T tokens
131k context
📦 CWM pretrained
Post-Training
Supervised Fine-tuning
Instruction and Reasoning
100B tokens
32k context
📦 CWM sft
Joint Reinforcement Learning
Agentic and Reasoning
172B tokens
131k context
📦 CWM
這種方法也用於 Phi-4-Mini-Reasoning ([Xu et al., 2025](https://arxiv.org/abs/2504.21233)) 的訓練,但有一個轉折:作者沒有在網頁資料上進行持續預訓練,而是使用從 DeepSeek-R1 蒸餾的推理 tokens 用於中期訓練語料庫。結果令人信服,顯示透過多階段訓練的一致和大幅收益:
模型 | AIME24 | MATH-500 | GPQA Diamond
---|---|---|---
**Phi-4-Mini** | 10.0 | 71.8 | 36.9
+ Distill Mid-training | 30.0 | 82.9 | 42.6
+ Distill Fine-tuning | 43.3 | 89.3 | 48.3
+ Roll-Out DPO | 50.0 | 93.6 | 49.0
+ RL (Phi-4-Mini-Reasoning) | **57.5** | **94.6** | **52.0**

這些結果促使我們嘗試類似的方法。從我們在 Open-R1 中創建和評估推理資料集的先前經驗中,我們有三個主要候選者可以使用:
  - [**Mixture of Thoughts**](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) **:** 從 DeepSeek-R1 跨數學、程式碼和科學蒸餾的 350k 推理樣本。
  - [**Llama-Nemotron-Post-Training-Dataset:**](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) NVIDIA 從各種模型(如 Llama3 和 DeepSeek-R1)蒸餾的大規模資料集。我們過濾了 DeepSeek-R1 輸出的資料集,產生了約 3.64M 樣本或 18.7B tokens。
  - [**OpenThoughts3-1.2M:**](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) 最高品質的推理資料集之一,包含從 QwQ-32B 蒸餾的 1.2M 樣本,包含 16.5B tokens。
由於我們計劃在最終 SFT 混合中包含推理資料,我們決定將 Mixture of Thoughts 保留用於該階段,其他用於中期訓練。我們使用 ChatML 作為聊天模板,以避免過早「燒入」SmolLM3 的模板。我們還使用 2e-5 的學習率訓練 5 個 epochs,使用 8 個節點以 128 的有效批次大小加速訓練。
何時進行中期訓練?
你可能想知道為什麼我們在進行了一些 SFT 執行_之後_討論中期訓練。按時間順序,中期訓練在基礎模型上的 SFT 之前發生。但進行中期訓練的決定只有在你執行了初始 SFT 實驗並識別出效能差距後才變得清晰。在實踐中,你經常會迭代:執行 SFT 以識別薄弱領域,然後進行有針對性的中期訓練,然後再次執行 SFT。將本節視為**「當 SFT 單獨不夠時該怎麼做」**。
**熔化 GPU 之謎**
在我們的叢集上執行這些實驗結果是一個令人驚訝的挑戰:老化的 GPU 會在各個點受到節流,這會導致硬體故障和每次執行的強制重新啟動。為了讓你了解情況,以下是其中一次執行的日誌,其中每種顏色代表一次重新啟動:
![Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/GtU8DnoWsAAruEG_28e1384e-bcac-8051-8122-ed6cacf8f632.AUNwy38i_Z1Unntg.webp)
我們最初認為 DeepSpeed 可能是罪魁禍首,因為加速器高度優化以提高輸送量。為了測試這一點,我們切換到 DP,這在某種程度上有所幫助,但隨後損失明顯不同!
在午夜在你的程式碼中發現錯誤比你想像的更常見。事後看來,對於這種規模的長期執行,使用 nanotron 會更有意義,因為它經過了實戰測試並且具有更快的輸送量。
![Image](https://huggingfacetb-smol-training-playbook.hf.space/_astro/Screenshot_2025-10-01_at_11_31_19_28e1384e-bcac-8005-8c5e-f0af3bf70372.BuE895O6_Z1ti3sF.webp)
正如我們後來發現的,Accelerate 中 DP 的一個錯誤意味著權重和梯度以模型的原生精度(在這種情況下為 BF16)儲存,這導致數值不穩定性以及在累積和優化期間梯度準確性的損失。
為了防止這種情況,大多數加速器對「主權重」和優化器狀態使用 FP32,並且僅在前向和後向傳遞中回傳到 BF16。
因此,我們切換回 DeepSpeed 並新增了積極的檢查點,以最小化 GPU 過熱和「從匯流排上掉下來」損失的時間。這種策略被證明是成功的,也是我們更普遍推薦的:
規則
正如我們在預訓練中強調的,在訓練執行期間經常儲存模型檢查點,理想情況下將它們推送到 Hugging Face Hub 以避免意外覆蓋。此外,使你的訓練框架對失敗具有穩健性,並能夠自動重新啟動。這兩種策略都將為你節省時間,特別是對於像中期訓練這樣的長執行作業。
在照看執行一週左右之後,我們終於有了結果:
總體而言,我們發現 NVIDIA 的後訓練資料集比 OpenThoughts 提供了更好的效能,但組合是最好的整體。
現在讓我們看看取這些檢查點之一並應用我們相同的基線資料混合的效果:
使用中期訓練推理模型而不是預訓練模型的效果是顯著的:使用擴展思考,我們幾乎將 AIME25 和 LiveCodeBench v4 的效能提高了三倍,而 GPQA-D 獲得了完整的 10 分增益。令人驚訝的是,推理核心也部分轉化為 `/no_think` 推理模式,在推理基準測試上提高了約 4-6 分。這些結果給了我們明確的證據,對於推理模型,如果你的基礎模型在預訓練期間還沒有看到大量推理資料,幾乎總是有必要進行一定數量的中期訓練。
何時不進行中期訓練
當你的模型必須學習新的核心技能時,中期訓練會發光。當基礎模型已經具有該技能,或者如果你試圖引發淺層能力(如風格或對話閒聊)時,它就不太有用。在這些情況下,我們建議跳過中期訓練,並將你的計算分配給其他方法,如偏好優化或強化學習。
一旦你對 SFT 資料混合和模型的廣泛能力有信心,焦點自然會從學習技能轉向完善它們。在大多數情況下,最有效的前進方式是_偏好優化_。
