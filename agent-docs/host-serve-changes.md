# 本次修改

在 RTX 3090 上不使用 Docker，直接把 Qwen3.8-27B EXL3 + DFlash2 的 OpenAI 相容服務跑起來，並讓長思考不會無限打轉、也不會把連線或答案掐壞。

服務預設聽 `http://0.0.0.0:8889`。

## 怎麼啟動

```bash
bash scripts/serve-host.sh
bash scripts/serve-host.sh --help
```

第一次裝環境與權重：

```bash
bash scripts/bootstrap-host.sh
```

`scripts/serve-host.sh` 會用專案裡的 `env/`，載入

- 主模型 `models/qwen38-27b-exl3`
- 草稿 `models/dflash2-exl3`
- 引擎 `exllamav3`（`r0b0tlab/exllamav3` `community` @ `355c6ee`，本機編成 sm_86）

PyTorch 沒有再裝一份。`env/` 透過 `reuse-torch.pth` 共用 `/home/sam/openwebui/.venv` 裡的 torch 2.13.0+cu130。映像檔測過的是 torch 2.10.0+cu130。升級那份 torch 之後要重跑 `bootstrap-host.sh`，把擴充套件重新編譯。

3090 的執行參數維持不變：`EXL3_HGEMM_F16ACC=auto`、`EXL3_INT8_GEMV=2`、`EXL3_INT8_GEMV_MAX_K=5`、`EXL3_QC_STAGING=1`，上下文 262144，KV cache 3-bit。

## 改了哪些檔

| 檔案 | 做了什麼 |
|---|---|
| `scripts/serve-host.sh` | 本機啟動腳本，含 `--help` 與下面的環境變數 |
| `scripts/bootstrap-host.sh` | 一次安裝：clone 引擎、建 venv、編譯 sm_86、下載權重 |
| `scripts/serve_openai.py` | 服務本體。串流、日誌、取消、取樣懲罰、思考收束都在這裡 |
| `container/serve_openai.py` | 與上面那份保持相同，給之後重建映像用 |
| `tests/test_think_exit.py` | 思考收束加分、以及「只在思考中罰重複字」的單元測試 |

## 1. 串流

`/v1/chat/completions` 和 `/v1/completions` 在 `"stream": true` 時改走 Server-Sent Events，HTTP chunked。

聊天片段是 `chat.completion.chunk`：

- 先送 `role`
- 思考中的文字走 `delta.reasoning_content`
- `</think>` 之後走 `delta.content`
- 最後一個片段帶 `finish_reason`、`usage`、`stats`
- 然後 `data: [DONE]`

不帶 `stream` 時仍是整段 JSON。這個模型預設會思考。短回答可以傳 `"chat_template_kwargs": {"enable_thinking": false}`。

## 2. 終端機日誌

跑 `scripts/serve-host.sh` 的那個終端機會看到每一則請求：

```text
[gen] 9df281cf 開始 max_tokens=32768 thinking=on dry=0.2 rep=1.0
[gen] 9df281cf 128 tok  86.4 tok/s  dflash acc=5.21 accept=103 rej=22  思考中
[gen] 9df281cf 結束：正常結束  6784 tok  70.4 tok/s  dflash acc=2.51 accept=4078 rej=14864  回答中
```

- `tok/s` 是解碼速度，不含 prefill
- `acc` 是 DFlash acceptance length（每輪驗證平均產出幾個 token）
- `accept` / `rej` 是草稿被接受和被退回的 token 數
- 生成超過約一秒才會印中間那行；結束行一定有

結束原因：

| 日誌 | 意思 |
|---|---|
| 正常結束 | 模型自己停，回答寫完 |
| 思考中碰到 max_tokens，被截斷 | 還在 `</think>` 之前就撞上上限 |
| 回答中碰到 max_tokens，被截斷 | 思考已結束，回答寫到一半 |
| 模型在思考中自行停止，沒有寫出回答 | 模型自己停了，但沒有回答 |
| 客戶端中止 | 前端把這條 HTTP 連線斷掉 |
| 思考內容重複打轉，已停止 | 只有在 `LOOP_WINDOW>0` 時才會硬停 |

同一份數字也在回應 JSON 的 `stats` 裡。

## 3. 預設 max_tokens

客戶端沒帶 `max_tokens` 時，聊天最多新生成 **32768** token。思考和回答都算在裡面。上下文仍是 262144。`/v1/completions` 的預設仍是 256。

客戶端有送 `max_tokens` 就用客戶端的值。

## 4. 前端按停止

先前生成跑在背景執行緒，連線斷了也不會停。現在 HTTP 執行緒發現客戶端關掉連線（或寫不出去）就設取消旗標，生成在下一個解碼步驟停掉，日誌是「客戶端中止」。

前端必須真的中斷這條連線（例如 `AbortController`）。只把畫面停掉、連線還掛著，後端看不出來。Prefill 那一步要等它跑完才會看到取消。

## 5. 不要用硬停來處理打轉

一度做過「偵測到同一段話就結束這次生成」。那個做法不對，預設已經關掉（`LOOP_WINDOW=0`）。打轉改由取樣懲罰和思考收束處理，請求不會因此被掐掉。

`LOOP_WINDOW` 大於 0 時才會回到引擎的 token 迴圈硬停（視窗與重複次數，同 ExLlamaV3 `chat.py`）。

## 6. 現在的懲罰

載入完成時會印這行，數字就是目前預設：

```text
[serve] READY  context=262144  default_max_tokens=32768  loop=off  dry=0.2 rep=1.0 freq=0.0/512  think_budget=4096+1536 think_freq=0.2
```

| 變數 | 預設 | 作用 |
|---|---|---|
| `DRY_MULTIPLIER` | 0.2 | 輕輕壓「整段照抄」。0 關掉 |
| `DRY_ALLOWED_LENGTH` | 6 | 短於等於這個長度的重複不罰，方法呼叫列表可以照常寫 |
| `DRY_BASE` | 1.75 | 重複越長，懲罰上升越快 |
| `DRY_RANGE` | 4096 | 往回看幾個 token。0 表示整段上下文 |
| `REP_PENALTY` | 1.0 | 單字重複懲罰。1.0 是關掉 |
| `FREQ_PENALTY` | 0 | 答案階段的頻率懲罰。預設關掉 |
| `FREQ_RANGE` | 512 | 上面那個懲罰的視窗，預設沒在用 |
| `THINK_FREQ` | 0.2 | 只在思考還沒結束時，懲罰最近用過的字 |
| `THINK_BUDGET` | 4096 | 思考超過這麼多新 token 後，`</think>` 的分數開始上升 |
| `THINK_RAMP` | 1536 | 從開始加分到加滿要再多少 token |
| `THINK_BIAS` | 16 | `</think>` 最多加多少 logit |

客戶端請求裡若帶了 `dry_multiplier`、`repetition_penalty`、`frequency_penalty`、`presence_penalty`、`max_tokens`，以客戶端為準。顯式傳 `0` 就是關掉那一項。

### 為什麼是這組數字

思考裡出現過一種打轉：句型固定成「需要可能提到「…」」，括號裡每次換一個詞。DRY 只罰完全相同的片段，罰不到這種換詞清單。

若把答案階段的頻率懲罰開到 0.3、DRY 開到 0.8，清單是壓住了，但回答裡重複的程式碼會被拆壞。實測 `$this->app->whenNot()` 會一路縮成 `$this()->whenNot()`、`$thi()->whenNot()`，段落寫不完。

所以改成：

- 答案階段不加頻率懲罰，DRY 只留 0.2，而且 6 個 token 以內的重複不罰。
- 思考還沒寫出 `</think>` 時，才用 `think_freq=0.2` 壓最近 256 個 token 裡反覆出現的字。答案一開始，這層就停。
- 思考超過 4096 個新 token 仍沒結束時，`</think>` 的 logit 線性加到最多 +16。模型轉去寫答案，連線不斷。

`tests/test_think_exit.py` 鎖住這兩件事：預算內不加分、超過後上升並封頂、提示或生成裡已經有 `</think>` 時不加；思考中的頻率懲罰只在思考還開著時降低重複 token，思考已結束則不動。

## 7. 用這題確認過

問題是「介紹關於 laravel 啟動順序與 service provider 的 register boot」。`temperature` 0，思考開著，客戶端不帶 `max_tokens` 也不帶懲罰欄位。連續兩次：

- `eos_reason` 是 `stop_token`，結束原因「正常結束」，階段是回答
- 答案從 `public/index.php` 講到 HTTP Kernel 與 bootstrappers
- 寫明先呼叫所有 Service Provider 的 `register()`，再呼叫 `boot()`。`register` 把服務綁進 container，`boot` 在 provider 都註冊之後才跑
- 思考裡沒有「需要可能提到」
- 答案裡沒有 `whenNot` 連寫，也沒有 `$thi()`

## 常用覆蓋

```bash
MAX_TOKENS=65536 bash scripts/serve-host.sh
THINK_BUDGET=8192 bash scripts/serve-host.sh
DRY_MULTIPLIER=0 bash scripts/serve-host.sh
```
