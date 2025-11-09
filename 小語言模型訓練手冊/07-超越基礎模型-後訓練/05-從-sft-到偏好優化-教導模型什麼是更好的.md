### [從 SFT 到偏好優化:教導模型什麼是_更好的_](https://huggingfacetb-smol-training-playbook.hf.space/#from-sft-to-preference-optimisation-teaching-models-what-better-means)

儘管你可以用更多資料繼續擴展 SFT,但在某個時候你會觀察到收益遞減或失敗模式,例如你的模型無法修復自己的錯誤程式碼。為什麼?因為 SFT 是一種_模仿學習_,所以模型只學習重現它訓練資料中的模式。如果資料還沒有包含良好的修復,或者如果所需的行為難以用蒸餾引發,模型就沒有明確的訊號來表明什麼算作「更好」。
即使你的資料集包含均勻混合的軌跡(即一些立即達到正確解決方案的軌跡,以及其他模型首先犯錯然後更正的軌跡),問題仍然存在。在這種情況下,模型可能只是學習犯初始錯誤是所需模式的一部分。當然,我們實際想要的是一個可以從一開始就產生正確解決方案的模型。
這就是偏好優化的用武之地。我們不只是複製示範,而是給模型比較回饋,如「回應 A 優於回應 B」。這些偏好為品質提供了更直接的訓練訊號,並使模型效能能夠擴展超出僅 SFT 的限制。
偏好優化的另一個好處是,你通常需要的資料遠少於 SFT,因為起點已經是一個相當不錯的模型,可以遵循指令並具有先前訓練階段的知識。
讓我們看看這些資料集是如何創建的。

#### [創建偏好資料集](https://huggingfacetb-smol-training-playbook.hf.space/#creating-preference-datasets)

從歷史上看,偏好資料集是透過向人類註釋者提供模型回應對並要求他們評分哪個更好(可能在一個量表上)來創建的。這種方法仍然被 LLM 提供者用於收集_人類偏好_標籤,但它非常昂貴並且擴展性差。最近,LLM 已經能夠產生高品質的回應,通常是以經濟實惠的方式。這些進步使 LLM 可以實際地為許多應用程式_生成_偏好。在實踐中,有兩種常見方法:
**強 vs. 弱**
  1. 取一組固定的提示 x(通常為覆蓋範圍和難度而策劃)。
  2. 從較弱或基線模型生成一個回應,從高效能模型生成另一個回應。
  3. 將較強模型的輸出標記為被選擇的回應 yc,將較弱的回應標記為被拒絕的 yr。
這產生了一個「較強 vs. 較弱」比較的資料集 ({x,yc,yr}),這很簡單構建,因為我們假設較強模型的輸出可靠地更好。
以下是來自 Intel 的流行範例,他們取了一個帶有來自 gpt-3.5 和 gpt-4 回應的 SFT 資料集,並透過選擇 gpt-4 回應作為被選擇的和 gpt-3.5 回應作為被拒絕的,將其轉換為偏好資料集:
**在策略上使用評分**
  1. 使用你將訓練的_相同模型_為相同提示生成多個候選回應。這創建了「在策略上」的資料,因為它反映了模型自然產生的輸出分布。
  2. 不是依賴較強模型作為參考,而是引入一個_外部評分者:_ 驗證器或獎勵模型,沿一個或多個品質軸(例如,幫助性或事實準確性)對回應評分。
  3. 然後評分者在候選回應中分配偏好標籤,產生更細緻和靈活的偏好資料集。
這種方法允許隨著模型改善而持續引導偏好資料,但其品質在很大程度上取決於評估者的可靠性和校準。
這樣的資料集的一個很好的範例來自 SnorkelAI,他們從流行的偏好資料集 [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) 中取得提示,將它們分成 3 組,然後迭代地應用上述配方來改善他們的模型:
在 SmolLM3 開發時,不存在任何帶有推理軌跡的偏好資料,因此我們決定使用「強 vs 弱」方法生成一些自己的資料。我們使用來自 Ai2 的 Tulu 3 偏好混合的提示,在 `/think` 模式下從 Qwen3-0.6B 和 Qwen3-32B 生成回應。結果是一個[大規模的 250k+ LLM 生成偏好資料集](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2/viewer/Preference/tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think),準備使用偏好優化演算法同時在多個軸上改善我們的 SFT 檢查點。

#### [我選擇哪個演算法?](https://huggingfacetb-smol-training-playbook.hf.space/#which-algorithm-do-i-pick)

Direct Preference Optimization (DPO) ([Rafailov et al., 2024](https://arxiv.org/abs/2305.18290)) 是第一個在開源中獲得廣泛採用的偏好優化演算法。
當 DPO 論文在 2023 年中期出版時,網上對它是否可以匹配 RL 方法存在激烈辯論,並且沒有顯示其在工業環境中有效性的配方。為了解決這個問題,我們在幾個月後發布了 [Zephyr 7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta),在合成資料上訓練整個模型,並展示了 DPO 的顯著效能增益。
它的吸引力來自於實作簡單、實踐中穩定,即使使用適度數量的偏好資料也有效。因此,DPO 已成為在使用更複雜的技術(如 RL)之前改善 SFT 模型的預設方法。
但研究人員很快發現有許多方法可以改善 DPO,如今有各種各樣的替代方案可供探索。下面我們列出了一些我們發現最有效的方法:
  - **Kahneman-Tversky Optimisation (KTO) [** [Ethayarajh et al. (2024)](https://arxiv.org/abs/2402.01306) **]:** KTO 不依賴偏好對,而是使用人類決策的想法來建模個別回應是否「可取」。如果你無法接觸配對偏好資料(例如從終端使用者收集的原始回應,如 👍 或 👎),這是一個好選擇。
  - **Odds Ratio Preference Optimisation (ORPO) [** [Hong et al. (2024)](https://arxiv.org/abs/2403.07691) **]:** 透過向交叉熵損失新增賠率比,直接將偏好優化整合到 SFT 中。因此不需要參考模型或 SFT 階段,這使得這種方法在計算上更有效率。
  - **Anchored Preference Optimisation (APO) [** [D'Oosterlinck et al. (2024)](https://arxiv.org/abs/2408.06266) **]:** 這是一個更可控的目標,明確正則化模型對被選擇 vs. 被拒絕輸出的可能性應該轉移多少,而不僅僅是優化它們的差異。有兩個變體(APO-zero 和 APO-down),其選擇取決於你的模型和偏好資料之間的關係,即被選擇的輸出是否優於模型或更差。
幸運的是,這些選擇中的許多只是 TRL 的 `DPOTrainer` 中的一行更改,因此對於我們的初始基線,我們執行了以下操作:
  - 使用來自 Ai2 的 [Tülu3 Preference Personas IF dataset](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following) 的提示和完成,以衡量在 `/no_think` 推理模式下對 IFEval 的指令跟隨的改善。
  - 重新使用上述提示,但現在使用 Qwen3-32B 和 Qwen3-0.6B 生成「強 vs. 弱」偏好對。這為我們提供了 `/think` 推理模式的偏好資料。
  - 訓練 1 個 epoch 並衡量 IFEval 上的_域內_改善,以及對其他評估(如 AIME25)的_域外_影響,這些評估與指令跟隨直接相關。
如下圖所示,兩種推理模式的域內改善都很顯著:在 IFEval 上,APO-zero 比 SFT 檢查點提高了 15-20 個百分點!
由於 APO-zero 還具有最佳的整體域外效能,我們決定在我們的其餘消融實驗中使用它。
偏好優化適用於推理
正如我們上面的結果所示,偏好優化不僅使模型更有幫助或對齊,它教它們_更好地推理_。如果你需要一種快速改善推理模型的方法,試著生成強 vs. 弱偏好並消融不同的損失函式:你可能會發現比普通 DPO 有顯著收益!

#### [哪些超參數對偏好優化最重要?](https://huggingfacetb-smol-training-playbook.hf.space/#which-hyperparameters-matter-most-for-preference-optimisation)

對於偏好優化,通常只有三個超參數會影響訓練動態:
  - 學習率,通常比用於 SFT 的學習率小 10-100 倍。
  - β 參數,通常控制偏好對之間的邊界大小。
  - 批次大小。
讓我們看看這些如何為 SmolLM3 產生結果,從我們在整個 `smoltalk2` 上訓練的 [SFT 檢查點](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints/tree/it-SFT)開始。
**使用小學習率以獲得最佳效能**
我們執行的第一個消融實驗是檢查學習率對模型效能的影響。我們執行實驗以確定學習率在比 SFT 學習率(2e-5)小約 200 倍(1e-7)到小約 2 倍(1e-5)之間的影響。像 Zephyr 7B 這樣的先前專案教會我們,偏好優化方法的最佳學習率大約比用於 SFT 的學習率小 10 倍,我們為 SmolLM3 執行的消融實驗證實了這個經驗法則。
如下圖所示,小約 10 倍的學習率在兩種推理模式下都改善了 SFT 模型的效能,但超過該 10 倍限制的所有學習率都會導致擴展思考模式的更差效能:
1e-75e-71e-65e-61e-501020304050Learning rateScore (%)
Legend
/think/no_thinkSFT checkpoint
MetricAIME25 GPQA Diamond IF-Eval LiveCodeBench v4 Average
`/no_think` 推理模式的趨勢更穩定,最佳學習率為 5e-6。這主要由單個基準測試(LiveCodeBench v4)驅動,因此我們在 SmolLM3 執行中選擇了 1e-6。
我們對你的訓練執行的建議是在比你的 SFT 學習率小 5 倍到 20 倍的範圍內執行學習率掃描。你很可能會在該範圍內找到最佳效能!
**調整你的 β**
我們為 β 參數執行的實驗範圍從 0.01 到 0.99,以探索鼓勵對參考模型不同程度對齊的值。作為提醒,較低的 beta 值鼓勵保持接近參考模型,而較高的值允許模型更密切地匹配偏好資料。β=0.1 的模型效能對於兩種推理模式都是最高的,並且與 SFT 檢查點的指標相比有所改善。使用低 beta 值會損害模型效能,並導致比 SFT 檢查點更差的模型,而在沒有擴展思考的情況下,效能在多個 β 值之間保持穩定。
這些結果表明,大於 0.1 的值對於偏好優化是優選的,並且將模型與偏好資料對齊比保持接近參考模型更有益。然而,我們建議在 0.01 和 0.5 範圍內探索 β 值。較高的值可能會抹去我們可能未在圖上顯示的評估中捕獲的 SFT 檢查點的能力。
0.010.050.100.501.0001020304050BetaScore (%)
Legend
/think/no_thinkSFT checkpoint
MetricAIME25 GPQA Diamond IF-Eval LiveCodeBench v4 Average
**擴展偏好資料**
我們還執行實驗以確定資料集大小如何影響結果,測試從 2k 到 340k 偏好對的值。在這個範圍內,效能保持穩定。擴展思考模式的效能下降發生在超過 100k 偏好對的資料集上,但下降不像我們在不同學習率值中看到的那麼明顯。我們用於 SmolLM3 訓練執行的資料集是 169k 偏好對,但結果顯示較小的資料集也顯示出比 SFT 檢查點的改善。對於未來的專案,我們知道我們可以在迭代階段嘗試較小的資料集,因為嘗試多個想法並快速識別最有前途的配置很重要。
2k10k20k100k200k01020304050Dataset sizeScore (%)
Legend
/think/no_thinkSFT checkpoint
MetricAIME25 GPQA Diamond IF-Eval LiveCodeBench v4 Average
**將所有內容整合在一起**
將所有這些線程整合在一起產生了最終的 SmolLM3-3B 模型:在其大小中最好的,並與 Qwen 自己的混合推理模型一起坐在 Pareto 前沿。
沒有推理的指令模型
2.02.53.03.54.04.50.00.51.01.52.02.53.03.54.04.55.0Model Size (Billion parameters)Win rate (%) • 8 popular LLM Benchmarksfaster / cheaperbetter
![](https://huggingfacetb-smol-training-playbook.hf.space/data/qwen-logo.svg)
Qwen31.7B
![](https://huggingfacetb-smol-training-playbook.hf.space/data/qwen-logo.svg)
Qwen2.5 3BInstruct
![](https://huggingfacetb-smol-training-playbook.hf.space/data/meta-logo.svg)
Llama3.2 3BInstruct
![](https://huggingfacetb-smol-training-playbook.hf.space/data/hf-logo.svg)
SmolLM33B
![](https://huggingfacetb-smol-training-playbook.hf.space/data/qwen-logo.svg)
Qwen34B
對於幾週的工作來說還不錯!

#### [參與規則](https://huggingfacetb-smol-training-playbook.hf.space/#rules-of-engagement-4)

總結我們關於偏好優化的發現,這些發現可能對你未來的專案有用:
  - 不要害怕創建你自己的偏好資料!隨著推理變得「太便宜而無法計量」,如今從各種[推理提供者](https://huggingface.co/docs/inference-providers/en/index)生成 LLM 偏好簡單且經濟實惠。
  - 選擇 DPO 作為你的初始基線並從那裡迭代。我們發現,根據偏好資料的類型,其他演算法(如 ORPO、KTO 或 APO)可以提供比 DPO 顯著的收益。
  - 使用比用於 SFT 的學習率小約 10 倍的學習率。
  - 掃描 β,通常在 0.01 到 0.5 範圍內
  - 由於大多數偏好演算法在一個 epoch 後過擬合,請分割你的資料並迭代訓練以獲得最佳效能。
偏好優化通常是簡單性和效能之間的最佳點,但它仍然繼承了一個關鍵限制:它只與你可以收集的離線偏好資料一樣好。在某個時候,靜態資料集用完訊號,你需要可以在模型與提示和環境互動時在線生成新鮮訓練回饋的方法。這就是偏好優化與更廣泛的_在策略和基於 RL 的方法_系列相遇的地方。
