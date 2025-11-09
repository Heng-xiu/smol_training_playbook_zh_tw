## [訓練馬拉松](https://huggingfacetb-smol-training-playbook.hf.space/#the-training-marathon)

你已經走到這一步了,恭喜!真正的樂趣即將開始。到這個時候,我們已經準備好了一切:一個經過驗證的架構、一個最終確定的資料混合,以及調整好的超參數。唯一剩下的就是設定基礎設施並按下「訓練」。對於 SmolLM3,我們在 384 個 H100 GPU(48 個節點)上訓練了將近一個月,處理了 11 兆 tokens。本節將引導你了解長期訓練執行期間實際發生的事情:飛行前檢查、不可避免的驚喜,以及我們如何保持穩定。你將親眼看到為什麼穩健的消融實驗實踐和可靠的基礎設施都很重要。我們在[最後一章](https://huggingfacetb-smol-training-playbook.hf.space/#infrastructure---the-unsung-hero)中介紹了 GPU 硬體、儲存系統和優化輸送量的技術基礎設施細節。我們的團隊已經經歷過這個過程很多次:從 StarCoder 和 StarCoder2,到 SmolLM、SmolLM2,現在是 SmolLM3。每一次執行都不同。即使你已經訓練了十幾個模型,每次新的執行都會找到一種新的方式讓你驚訝。本節是關於將勝算堆疊到你這一邊,以便你為這些驚喜做好準備。

