### [SmolLM3:策劃資料混合(網頁、多語言、數學、程式碼)](https://huggingfacetb-smol-training-playbook.hf.space/#smollm3-curating-the-data-mixture-web-multilingual-math-code)

對於 SmolLM3,我們希望有一個能夠處理英語和多種其他語言,並在數學和程式碼方面表現出色的模型。這些領域 — 網頁文本、多語言內容、程式碼和數學 — 在大多數 LLM 中很常見,但我們將在這裡描述的過程同樣適用於你正在訓練低資源語言或特定領域(如金融或醫療保健)的情況。方法是相同的:識別良好的候選資料集,執行消融實驗,並設計一個平衡所有目標領域的混合。我們不會在這裡介紹如何建立高品質資料集,因為我們已經在早期工作(FineWeb、FineWeb2、FineMath 和 Stack-Edu)中詳細說明了這一點。相反,本節重點介紹我們如何將這些資料集_組合_成一個有效的預訓練混合。

#### [**建立在經過驗證的基礎上**](https://huggingfacetb-smol-training-playbook.hf.space/#building-on-proven-foundations)

當涉及到預訓練資料時,好消息是我們很少需要從頭開始。開源社群已經為大多數常見領域建立了強大的資料集。有時我們需要創造新的東西 — 就像我們用 Fine 系列(FineWeb、FineMath 等)所做的那樣 — 但更多時候,挑戰在於選擇和組合現有來源,而不是重新發明它們。這就是我們在 SmolLM3 中的情況。SmolLM2 已經在 1.7B 參數下為英語網頁資料建立了一個強大的配方,並識別了我們能接觸到的最佳數學和程式碼資料集。我們的目標是將這種成功擴展到 3B,同時增加某些能力:穩健的多語言性、更強的數學推理和更好的程式碼生成。

#### [**英語網頁資料:基礎層**](https://huggingfacetb-smol-training-playbook.hf.space/#english-web-data-the-foundation-layer)

網頁文本構成了任何通用 LLM 的骨幹,但品質與數量同樣重要。從 SmolLM3 來看,我們知道 FineWeb-Edu 和 DCLM 是訓練時最強大的開放英語網頁資料集。它們一起為我們提供了 5.1T tokens 的高品質英語網頁資料。問題是:最佳混合比例是多少?FineWeb-Edu 有助於教育和 STEM 基準測試,而 DCLM 則改善常識推理。遵循 SmolLM2 方法,我們在 100B tokens 上對我們的 3B 模型執行了掃描,測試了 20/80、40/60、50/50、60/40 和 80/20(FineWeb-Edu/DCLM)的比例。混合它們(大約 60/40 或 50/50)給出了最佳權衡。我們在 100B tokens 上訓練的 3B 模型上重新執行了與 [SmolLM2 論文](https://arxiv.org/abs/2502.02737v1)相同的消融實驗,並得出了相同的結論。使用 60/40 或 50/50 在各基準測試中提供了最佳平衡,與我們的 SmolLM2 發現相符。我們為階段 1 使用了 50/50 比例。我們還新增了其他資料集,如 [Pes2o](https://huggingface.co/datasets/allenai/dolmino-mix-1124/tree/main/data/pes2o)、[Wikipedia & Wikibooks](https://huggingface.co/datasets/allenai/dolmino-mix-1124/tree/main/data/wiki) 和 [StackExchange](https://huggingface.co/datasets/HuggingFaceTB/stackexchange_2025_md),這些資料集對效能沒有任何影響,但我們包含它們以改善多樣性。

#### [**多語言網頁資料**](https://huggingfacetb-smol-training-playbook.hf.space/#multilingual-web-data)

對於多語言能力,我們針對其他 5 種語言:法語、西班牙語、德語、義大利語和葡萄牙語。我們從 FineWeb2-HQ 中選擇了它們,總共為我們提供了 628B tokens。我們還以較小的比例包含了其他 10 種語言,如中文、阿拉伯語和俄語,不是為了針對它們達到最先進的效能,而是為了讓人們能夠輕鬆地對 SmolLM3 進行持續預訓練。我們對 FineWeb2-HQ 中不支援的語言使用了 FineWeb2。關鍵問題是:我們的網頁資料應該有多少是非英語的?我們知道模型在某種語言或領域中看到的資料越多,它在該語言或領域中就越好。權衡來自我們固定的計算預算:增加一種語言的資料意味著減少其他語言(包括英語)的資料。透過在 3B 模型上的消融實驗,我們發現網頁混合中 12% 的多語言內容達到了正確的平衡,在不降低英語基準測試的情況下改善了多語言效能。這符合 SmolLM3 的預期用途,其中英語仍將是主要語言。還值得注意的是,只有 628B tokens 的非英語資料與 5.1T 英語 tokens 相比,增加更多會需要對多語言資料進行更多重複。

#### [**程式碼資料**](https://huggingfacetb-smol-training-playbook.hf.space/#code-data)

我們階段 1 的程式碼來源是從 [The Stack v2 和 StarCoder2](https://arxiv.org/abs/2402.19173) 訓練語料庫中提取的:
  - [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)(16 種語言)作為我們的基礎,經過 StarCoder2Data 的過濾。
  - StarCoder2 GitHub pull requests 用於真實世界的程式碼審查推理。
  - Jupyter 和 [Kaggle notebooks](https://huggingface.co/datasets/HuggingFaceTB/issues-kaggle-notebooks) 用於可執行的、逐步的工作流程。
  - [GitHub issues](https://huggingface.co/datasets/HuggingFaceTB/issues-kaggle-notebooks) 和 [StackExchange](https://huggingface.co/datasets/HuggingFaceTB/stackexchange_2025_md) 討論串用於圍繞程式碼的上下文討論。
[Aryabumi et al. (2024)](https://arxiv.org/abs/2408.10914) 強調程式碼在程式設計之外改善了語言模型的效能,例如在自然語言推理和世界知識方面,並建議在訓練混合中使用 25% 的程式碼。受此啟發,我們以 25% 的程式碼在混合中開始我們的消融實驗。然而,我們觀察到英語基準測試(HellaSwag、ARC-C、MMLU)上的顯著降低。減少到 10% 程式碼,與 0% 程式碼相比,我們沒有看到英語基準測試套件的改善,但我們還是包含了它,因為程式碼是模型中一個非常重要的能力。我們延遲新增 Stack-Edu — 我們對 StarCoder2Data 進行教育過濾的子集 — 直到後期階段,遵循將高品質資料分階段以獲得最大後期訓練影響的原則。

#### [**數學資料**](https://huggingfacetb-smol-training-playbook.hf.space/#math-data)

數學遵循與程式碼類似的哲學。早期,我們使用了較大、更通用的集合 FineMath3+ 和 InfiWebMath3+,後來我們對 FineMath4+ 和 InfiWebMath4+ 進行了上採樣,並引入了新的高品質資料集:
  - MegaMath ([Zhou et al., 2025](https://arxiv.org/abs/2504.02807))
  - 指令和推理資料集,如 OpenMathInstruct ([Toshniwal et al., 2024](https://arxiv.org/abs/2402.10176)) 和 OpenMathReasoning ([Moshkov et al., 2025](https://arxiv.org/abs/2504.16891))
我們在階段 1 中使用 3% 的數學,在 FineMath3+ 和 InfiWebMath3+ 之間平均分配。只有 54B tokens 可用,估計 8T 到 9T token 的階段 1,使用超過 3% 的數學將需要在資料集上進行超過 5 個 epochs。

#### [**為新階段找到正確的混合**](https://huggingfacetb-smol-training-playbook.hf.space/#finding-the-right-mixture-for-new-stages)

雖然我們從頭執行消融實驗以確定階段 1 混合,但為了測試新階段(在我們的情況下是兩個新階段)的新資料集,我們使用了退火消融實驗:我們在大約 7T tokens(階段 1 後期)取一個檢查點,並執行 50B token 退火實驗,設置如下:
  - **40% 基線混合**:我們一直在訓練的確切階段 1 混合
  - **60% 新資料集**:我們想要評估的候選資料集
例如,為了測試 MegaMath 是否會改善我們的數學效能,我們執行了 40% 階段 1 混合(維持 75/12/10/3 領域分割)和 60% MegaMath。可以在下一節中找到 3 個階段的組成。隨著我們的資料經過仔細策劃,我們的混合透過消融實驗得到驗證,我們準備開始實際的訓練之旅。接下來的章節是 SmolLM3 為期一個月的訓練執行的故事:準備工作、意外挑戰以及沿途學到的教訓。
