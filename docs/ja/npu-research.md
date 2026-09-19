# AMD Ryzen AI NPU での超解像: 研究ノート

AMD Ryzen AI の NPU（XDNA2）で超解像モデルを動かし、同じチップの iGPU と速度・画質・
リソース占有で比較した記録。2026-08 から 2026-09 の測定をトピック別にまとめている。
数値はすべて 1 台の実機での実測で、推測は推測と明記する。

英語の個別ページ: [tail-cut](../tail-cut.md)、[SwinIR を NPU で動かす](../swinir-npu.md)、[AdcSR を NPU で動かす](../adcsr-npu.md)。
実装は Windows 用アップスケーラー [tomhda/ultraeasy-upscaler](https://github.com/tomhda/ultraeasy-upscaler) にある。

要点:

- bf16 + VAIML コンパイルで、fp32 とほぼ同じ画質の 4 倍超解像を iGPU をほぼ占有せずに実行できる。
- 軽量モデル（4xNomosUni SPAN、Anime Video v3）は、末尾の DepthToSpace 以降を CPU に出す
  構成（tail-cut）で NPU 時間が約半分になり、SPAN は同じチップの iGPU（DirectML）の約 2 倍速い。
- この NPU は演算のピーク性能より、大きなテンソルの出し入れと並べ替えが律速になる。
  Real-ESRGAN（RRDB）・SwinIR-M・AdcSR はこの制約で頭打ちになっている。

## 1. 環境と測定条件

- AMD Ryzen AI 7 PRO 350（NPU: XDNA2、8 列）/ Radeon 860M（iGPU）/ 32GB LPDDR5-8000
- Ryzen AI Software 1.8.0（conda env `ryzen-ai-1.8.0`、onnxruntime 1.27 VitisAI EP、XRT 2.19.0、
  NPU ドライバ 32.0.203.329）。2026-08-15 より前の節は Ryzen AI Software 1.7.1（Quark 0.11rc1）での測定。
  bf16 変換（Quark）は 1.7.1 の環境のものを使う。
- GPU 側: DirectML（`tools/winml-sr`、fp32 ONNX）。2026-08 上旬の節は realesrgan-ncnn-vulkan（NCNN fp16）。
- 検証素材: Big Buck Bunny（bbb、853x480）/ Tears of Steel（tos、1280x534）（CC-BY）、Superman 1941（PD）
- NPU の電源モードは、断りがない限り Default（`xrt-smi examine` の Power Mode）。
- 「忠実度」は同一モデルの fp32 出力との PSNR。40 dB 前後は目視でほぼ判別できない水準。

## 2. 現在の実測（2026-09-19）

入力 853x480 → 4 倍（3412x1920）、タイル分割・結合・色変換込みの 1 枚あたり（常駐セッションの定常値）。

| モデル | GPU（DirectML, fp32） | NPU（Default） | NPU（Turbo） | NPU 忠実度 |
|---|---|---|---|---|
| 4xNomosUni SPAN | 0.51 秒 | **0.25 秒** | 0.18 秒 | 46.9 dB |
| Anime Video v3 | 0.46 秒 | 0.47 秒 | 0.35 秒 | 49.4 dB |
| Real-ESRGAN（AMD縮小版 RRDB） | 2.78 秒 | 2.07 秒 | 1.23 秒 | 37.9 dB（1.7.1 時点） |
| SwinIR-M | 約 53 秒 | 約 79 秒 | 256 タイル 6.73 → 3.39 秒 | 38.5 dB |
| AdcSR | 約 1.3〜1.6 秒 / 128 タイル | 約 2.05 秒 / 128 タイル | 約 1.05 秒 / 128 タイル | 45.4 dB（GPU 版との比較） |

動画（rawvideo パイプライン、3 秒 72 フレームの E2E、Default）:
NPU × SPAN 3.61 fps、NPU × Anime Video v3 1.95 fps、GPU × Anime Video v3 2.48 fps。
Turbo では SPAN 4.89 fps、Anime Video v3 2.77 fps。

NPU 実行中の iGPU 3D エンジンは idle 水準、CPU は 2〜7%（5.5 の占有率の表）。

## 3. この NPU の性質（測定から分かったこと）

### 3.1 演算のピーク性能と実効の差

`xrt-smi validate` は 3 種すべて PASSED: gemm 51.3 TOPS（公称 INT8 50 TOPS と一致）、
latency 平均 67.5 us、throughput 平均 60889.2 ops。
超解像モデルの実効（SPAN で 238.8 GOPs / 0.25 秒 = 約 950 GOPS）はピークの約 1/50〜1/100 で、
律速は演算器のピークではない。

### 3.2 律速は大きなテンソルの出し入れと並べ替え

AI Analyzer（provider option `ai_analyzer_profiling` / `ai_analyzer_visualization`）の
AIE レイヤタイマをモデル別に読んだ結果:

| モデル | 内訳 |
|---|---|
| SPAN（512 タイル、約 240 ms） | 中盤約 20 レイヤが一律 2.0〜2.3 ms（計 42 ms）。終盤の upsampler（DepthToSpace・L3 spill 系）4 レイヤが 17/42/20/18 ms（計 97 ms）、続く区間 71 ms。終盤で約 7 割 |
| AdcSR 後半（VAE デコーダ、約 1400 ms） | AIE レイヤ時間の 92%（1023 ms / 1111 ms）が `L3_OFM_Buffer_spill`。attention・Softmax・MatMul は 2 レイヤ 2.6 ms |
| SwinIR-M（256 タイル、約 5.6 秒） | `L3_OFM_Buffer_spill` を含む区間が 451 個・計 2812 ms（50%）。重い層の詳細は 5.5 |
| Real-ESRGAN（RRDB、256 タイル約 170 ms） | 末尾ではなく全層が約 1 ms ずつ（5.4） |

- SPAN の中盤レイヤの一律約 2.1 ms は、サイズによらない固定費と推測する（レイヤ起動と重みの転送）。
- 大きな活性（数十 MB 級）を層ごとにメインメモリへ出し入れする時間が、演算より長い。
  DepthToSpace のような「並べ替えだけの演算」が特に遅い。

### 3.3 NPU 時間は処理ピクセル数にほぼ線形

Anime Video v3（PReLU 分解版、bf16）で約 2.0 µs/px、タイルごとの固定費は約 7 ms（2026-08、Windows ML 経由の測定）。
タイルサイズは「パディングとオーバーラップで捨てるピクセルが最小になる値」で選ぶ。
853x480 は core 480 の 512 タイル（縦がちょうど収まり 2 枚）で捨てピクセルが 28% まで減る。

フレーム全体（854x480）を 1 回で推論する構成は成立しない。VAIML コンパイルは通るが、
実行時に XRT の単一ディスパッチ時間制限を超える（ERT_CMD_STATE_TIMEOUT）。タイル分割は必須。

### 3.4 コンパイル設定と FP32 直接入力

- 同梱の既定設定（`vaip_config.json`）は optimize_level 2 相当。
  SPAN 512 タイル（bf16cast、基準 0.2457 秒）で比較すると、O1 は 1.36〜1.38 秒（約 5.6 倍遅い。コンパイルは速い）。
  O3・`preferred_data_storage` の vectorized / unvectorized は基準と同等で、出力が既存キャッシュと
  ビット一致するものもあった。既定のままが最良。
- FP32 ONNX の直接入力（コンパイル時に自動で bf16 化）はモデル依存。
  SPAN は bf16cast と同速（0.256 秒）で、fp32 参照比 PSNR が 44.51 → 46.94 dB に上がる（Quark 変換も不要）。
  AdcSR 後半は FP32 直接入力だと 6 サブグラフ・計 76 ノードしか NPU に載らず、残りが CPU で動いた。
  新しいモデルを NPU 化したら、キャッシュ内 `context.json` の metaDef が 1 個で全ノードを含むことを確認する。
- 同一グラフを再コンパイルすると実行時間が変わる。SwinIR の 2 ブロックモデルで 0.375 秒 / 0.406 秒（8% 差）。
  小さな改善を評価するときは、基準側も複数回コンパイルして比べる。

### 3.5 NPU の電源モード

`xrt-smi configure --pmode turbo`（AC 電源必須）に切り替え、同じキャッシュ・同じスクリプト
（warmup 2 + 10 run、SwinIR-M は 3 run）で測り直した。再コンパイルは不要。
出力は SPAN / Anime Video v3（tail-cut）・AdcSR とも Default とビット一致。

| モデル（1 run 中央値） | Default | Turbo | 比 |
|---|---|---|---|
| SPAN 512 タイル（全体モデル） | 0.246 秒 | 0.130 秒 | 1.89 倍 |
| Anime Video v3 512 タイル（全体モデル） | 0.524 秒 | 0.257 秒 | 2.04 倍 |
| AdcSR 前半（128 タイル） | 0.738 秒 | 0.361 秒 | 2.04 倍 |
| AdcSR 後半 | 1.359 秒 | 0.628 秒 | 2.16 倍 |
| SwinIR-M 256 タイル | 6.73 秒 | 3.39 秒 | 1.99 倍 |

AdcSR の 2 プロセス構成（overlap 32）:

| 条件 | Default | Turbo |
|---|---|---|
| tos 中央 256x256（16 タイル） | 35.5 秒（2.2 秒/タイル） | 16.9 秒（1.06 秒/タイル） |
| tos 1280x534（180 タイル） | 未計測（2.05〜2.1 秒/タイルから約 370 秒の見込み） | 188.7 秒（1.05 秒/タイル） |

NPU の消費電力（`xrt-smi examine` の Power の報告値。SwinIR-M 256 タイルの連続推論中に約 0.8 秒間隔で採取。外部計測ではない）:

| 状態 | Default | Turbo |
|---|---|---|
| NPU を使うプロセスが無いとき（約 0.9 秒間隔で読み続けた場合） | N/A（12 回中 12 回） | 1 回目 N/A、以後 0.28〜0.36 W |
| 同上（20〜45 秒触らずに置いた直後の 1 回目） | N/A | N/A（2 回目は 0.24〜0.27 W） |
| セッションの読み込み中（推論の前） | 0.04〜0.16 W | 0.1〜0.4 W |
| 推論中 | 0.6〜1.2 W（平均 約 0.85 W） | 2.4 W で一定 |
| 1 タイルの時間 | 6.55 秒 | 3.39 秒 |
| 1 タイルあたりの NPU の電力量（時間 × 平均電力） | 約 5.6 J | 約 8.1 J |

- Turbo では `xrt-smi examine` で読むこと自体が NPU を起こす（触らずに置くと N/A に戻る）。
  誰も読まない通常の待機中は、どちらのモードも電力が報告されない状態になる。
- Turbo は約 2 倍速いが、NPU の電力量は 1 タイルあたり約 1.45 倍。絶対値はどちらも数 W 未満。
  処理時間が半分になる分、CPU やメモリなど NPU 以外の消費を含めた合計がどうなるかは未計測。
- Turbo に設定したままバッテリー駆動に切り替えても、Power Mode は Turbo のままで、
  SwinIR-M は 3.49 秒/タイル・2.4 W（AC 接続時は 3.39 秒）。設定が自動で Default に戻ることはなかった。
- Default は Windows の電源モードに追従する設定（AMD 文書）。Default のときの実クロックと発熱は未計測。
  再起動後に設定が保持されるかは未確認。
- 電源モードは OS 側の設定で、アプリからは変更しない。公開している実測値は Default を基準にする。

### 3.6 GPU との同時実行は合計スループットが増えない

NPU と GPU（DirectML）で同時に別の画像を処理した（SwinIR-M、448x448・4 タイル、ヘルパー起動込み）。
単独は NPU 約 35 秒 / GPU 約 35 秒、同時実行はどちらも 60〜74 秒。合計スループットは逐次実行と変わらない。
タイルを NPU と GPU に分担させる構成は不採用。原因は未特定
（NPU と iGPU が同じ LPDDR5 を共有するため、メモリ帯域の競合と推測）。

### 3.7 長い推論と TDR ライブダンプ

1 回の推論が約 3 秒を超えると、Windows の TDR 検知によるカーネルライブダンプ採取が周期的に起きる
（WER バケット `LKD_0x141_Tdr:6_IMAGE_ipustack.sys`）。SwinIR-M（1 推論 6.6 秒）では約 38 秒周期
（タイル 6 枚ごと）で最大 +7GB の過渡的なコミット増を観測した。プールタグ vTDR/dxgkrnl → Ldmp/ntoskrnl の
シーケンスがスパイクと同期し、NPU アダプタの DXGI 共有メモリは変化しない。
採取中は約 0.7 秒システムが停止するが、推論自体はリセット・エラー・速度低下なし。
SwinIR-S（3.0 秒/タイル）でも発生した。Anime Video v3 / SPAN では発生しない。
AMD へ報告済み（[amd/RyzenAI-SW#398](https://github.com/amd/RyzenAI-SW/issues/398)）。

### 3.8 計測方法

- NPU 専用のパフォーマンスカウンタは無く、`GPU Engine` カウンタセットに別 LUID のデバイスとして現れる。
  値は推論 1 回につき 1 サンプルのバースト報告（約 650%）なので、中央値ではなく時間積分の平均で読む。
  `xrt-smi` で HW コンテキストの Active 状態と使用カラム数を確認できる。
- AI Analyzer の生成物は実行時のカレントディレクトリ（`record_timer_ts.json` ほか）とキャッシュ内
  （`vaiml_par_0/0/aie_record_timer.json`、`layer_name_map.json`）に出る。
  `record_timer_ts.json` は 1 GHz のタイムスタンプ列で、隣接する cycle の差が区間の時間。
  区間を後ろ側タイムスタンプの layer_id に帰属させ、`layer_name_map.json` の `user_friendly_name` で
  演算名に対応づける。プロファイル ON の時間は OFF の基準と比べない。

## 4. 精度形式: int8 と bf16

### 4.1 int8（XIR フロー）は Q/DQ と多サブグラフ分割が支配する

int8（XINT8/u8s8、XIR フロー）では、演算量が約 10 倍違う RRDB と SRVGGNetCompact が同速
（335 vs 323 ms/タイル）で、タイルを 256 → 512 に 4 倍化しても MP/s は変わらなかった（1.02 倍）。
約 4.2 MP/s の帯域律速に見えるが、実際は Q/DQ 変換と多サブグラフ分割（Real-ESRGAN で 11 個）の
オーバーヘッドが支配していた。劣化はエッジのギザギザとして現れる。

Ryzen AI 1.8 で既定になった整数バックエンド（provider option `target="X2"`、A8W8 / XINT8）は未検証。

### 4.2 bf16 + VAIML はキャリブレーション不要で int8 より速く高画質

`quark.onnx.tools.convert_fp32_to_bf16 --format with_cast` で fp32 ONNX を Cast ベースの bf16 に変換すると、
VitisAI EP が VAIML フローで単一サブグラフにコンパイルする。

| モデル | int8 → bf16 タイル速度 | fp32 忠実度（PSNR mean/min） |
|---|---|---|
| Real-ESRGAN（縮小 RRDB） | 335 → **168 ms**（2.0 倍） | 37.75/34.06 → **37.87/36.37 dB** |
| Real-ESRGAN Anime（RRDB 6B） | 351 → **262 ms**（1.3 倍） | 35.89/31.83 → **39.41/37.24 dB** |

目視でも GPU fp16 と NPU bf16 は区別できない（領域差分 35 dB 超）。
キャリブレーション画像、巨大な一時ファイル、長時間の探索がすべて不要になる。

`ModelQuantizer` の BF16 config は `com.amd.quark:ExtendedQuantizeLinear`（カスタム op）を出力し、
カスタム op ライブラリが無い環境では動かない。tools の変換スクリプト（標準 op のみ）を使う。

### 4.3 int8 キャリブレーションの記録（bf16 移行前）

- キャリブレーション素材のドメイン一致とクリーンさが効く。グレイン入り素材を除き、
  日本のアニメ + BBB の 32 枚に変えるだけで +0.4 dB（min +0.6 dB）。
- Quark MinMSE は %TEMP% に巨大な中間テンソルを書く（RRDB 256px で約 2.5GB/枚。512px × 100 枚では 116GB）。
- 64 枚 × RRDB は 32GB RAM で失敗（探索フェーズのメモリはほぼ枚数に比例）。

## 5. モデル別の記録

### 5.1 Anime Video v3（SRVGGNetCompact）: PReLU の誤コンパイルと分解

realesr-animevideov3 の bf16 は、Ryzen AI 1.7.1 の VAIML でエラーなくコンパイル・実行されるが
出力が発散した（PSNR −40 dB、max|diff| > 1500、決定論的）。4 段階の二分探索で原因を特定した。

| 段階 | 実験 | 結果 |
|---|---|---|
| 1 | 単体 op 5 種（PReLU / PixelShuffle / Resize+Add 等） | すべて正常 |
| 2 | 同構造を深さ 2/8/16・活性化 3 種（slope=0.25） | すべて正常 |
| 3 | 実重み / onnxsim の有無 / 合成 + onnxsim | 実重みのみ異常。onnxsim は無関係 |
| 4 | 合成 + 負 slope / 実重み + PReLU 分解 | 負 slope で忠実度 59 → 35 dB、コンパイル 38 → 1005 秒。分解版は正常（38.4 dB） |

原因は実モデルの PReLU slope（min −1.38 / max +1.69。負値と 1 超えを含む）。
完全な発散には実重みのもう一要素（巨大なバイアス等との複合）が関与するとみられる。

回避策（採用済み）: fp32 段階で `PReLU(x) = ReLU(x) − w⊙ReLU(−x)` に等価分解してから bf16 変換する
（`export_animevideov3.py --decompose-prelu`）。再現実験は `scripts/npu/bisect_vaiml_bf16*.py`。
1.8 系の VitisAI EP（Windows ML 配信版）では非分解モデルでも誤コンパイルは起きなかった（PSNR 59.7 dB）が、
分解版のほうが約 2 倍速い（256 タイルで 133 ms vs 274 ms）ため、分解は高速化としても有効。

### 5.2 tail-cut: 末尾の DepthToSpace 以降を CPU で実行（Anime Video v3 / 4xNomosUni SPAN）

3.2 の「終盤の upsampler が約 7 割」から、DepthToSpace（blocksize 4、CRD）の直前でグラフを切り、
NPU は `[1,48,512,512]` を出力、PixelShuffle（Anime Video v3 は入力の最近傍 4 倍の加算を追加）を
CPU の numpy で行う。出力の要素数は同じなので転送量は変わらない。
CPU 後処理は次タイルの NPU 実行と 2 スレッドで重ねる（キュー深さ 2）。
切断は `scripts/npu/split_tail.py`（CPU EP で元モデルとの一致を検証してから body とマニフェストを出力）。

1 枚あたり（512 タイル overlap 16、Default、導入時の測定）:

| モデル・入力 | 全体モデル | tail-cut | 全体版との PSNR |
|---|---|---|---|
| SPAN、bbb 853x480 | 0.608 秒 | **0.358 秒** | 46.91 dB |
| SPAN、tos 1280x534 | 1.692 秒 | **0.914 秒** | 46.48 dB |
| Anime Video v3、bbb 853x480 | 1.211 秒 | **0.615 秒** | 53.35 dB |
| Anime Video v3、tos 1280x534 | 3.504 秒 | **1.635 秒** | 54.56 dB |

- SPAN body は fp32 直接入力（73 ノード、238.776 GOPs、VAIML 対応 100%、サブグラフ 1）。
  fp32 参照比 PSNR は bf16cast 全体版の 44.51 dB → 46.94 dB。
  全体版との相互 PSNR 46.9 dB は bf16 の丸めの差で、劣化ではない。
- Anime Video v3 の body は PReLU 分解版の bf16cast（onnx 417 ノードのうち Cast 融合で 311 ノード、
  328.087 GOPs、サブグラフ 1）。
- CPU 後処理（SPAN 約 19 ms/タイル、Anime Video v3 約 57 ms/タイル）は NPU 実行より短く、重ねると隠れる。
- 動画（3 秒 72 フレームの E2E）: SPAN 1.57 → 2.44 fps、Anime Video v3 0.79 → 1.46 fps。
  NPU 実行中の CPU 使用率（全体、5 秒ごと）は新旧で同水準（10% 台前半。ffmpeg の伸縮・符号化が支配的）。
- `--tail` なしの 1 モデル経路は変更前とビット一致。body とマニフェスト（`*.tail.json`）が無い環境では
  全体モデルで動く。`UEU_NPU_TAILCUT=0` で無効化。

### 5.3 npu_serve の CPU 側の削減

tail-cut と Turbo で NPU 実行が縮むと、1 枚あたりの時間の 4〜5 割が CPU 側になった。
SPAN tail-cut・bbb 853x480（2 タイル）の内訳（変更前、Turbo）:
NPU 実行 計 約 145 ms、タイル結合 約 45 ms、`clip(x*255).astype(uint8)` 約 50 ms、CHW→HWC 約 20 ms
（いずれも全タイルの NPU 実行が終わった後に直列で実行）。
Anime Video v3 は `np.repeat` 2 回による最近傍拡大と加算が 1 タイル約 120 ms で、
NPU 実行（約 140 ms/タイル）とほぼ並んでいた。

変更:

- 量子化までをタイル単位にし、tail の後処理スレッドで実行する（次タイルの NPU 実行と重なる）。
- 先に有効コア（overlap と右端・下端のパディングを除いた領域）を低解像度側で切り出し、
  拡大後の float 全体と結合用の float バッファを作らない。
- pixel shuffle の並べ替えと uint8 化を、最終出力バッファ（HWC uint8）への 1 回の書き込みにまとめる。
  最近傍拡大は放送（`[h,1,w,1,c]`）で加算し、拡大済みの配列を作らない。
- 全体モデル経路（tail なし）も同じくタイル単位で量子化して直接書き込む。
- 応答は `tobytes()` を作らずに書き、クライアントは受信先の配列へ `readinto` する（中間コピー 3 回を削除）。

演算は要素ごとに従来と同じで、出力は SPAN / Anime Video v3（tail-cut）、Anime Video v3 全体モデル、
AdcSR の 2 プロセス構成で変更前とビット一致（sha256 で確認）。

| 条件（1 枚あたり） | Default 変更前 | Default 変更後 | Turbo 変更前 | Turbo 変更後 | GPU（DirectML） |
|---|---|---|---|---|---|
| SPAN、bbb 853x480 | 0.362 秒 | 0.250〜0.274 秒 | 0.275〜0.282 秒 | 0.173〜0.181 秒 | 0.51 秒 |
| SPAN、tos 1280x534 | 0.914 秒 | 0.677〜0.687 秒 | 0.678〜0.681 秒 | 0.448〜0.471 秒 | |
| Anime Video v3、bbb 853x480 | 0.615 秒 | 0.466〜0.494 秒 | 0.447〜0.455 秒 | 0.346〜0.351 秒 | 0.46 秒 |
| Anime Video v3、tos 1280x534 | 1.635 秒 | 1.324〜1.334 秒 | 1.128〜1.131 秒 | 0.879〜0.881 秒 | |

動画（3 秒 72 フレームの E2E）: Default で SPAN 2.44 → 3.61 fps（19.96 秒）、
Anime Video v3 1.46 → 1.95 fps（36.91 秒）。Turbo で SPAN 4.89 fps（14.73 秒）、Anime Video v3 2.77 fps（25.99 秒）。
SPAN bbb の残りは、Turbo で NPU 実行 約 145 ms に対し約 30 ms
（入力の float 化と分割、最後のタイルの後処理、約 20 MB のパイプ書き込み）。

### 5.4 Real-ESRGAN（AMD縮小版 RRDB）

bf16 の基本値は 4.2 の表。853x480 の 1 枚は NPU 2.07 秒 / GPU 2.78 秒（Default、5.3 の変更後）。

tail-cut の検討（不採用）: RRDB の末尾は DepthToSpace ではなく、最近傍 Resize × 2 と 32ch の Conv 4 本
（うち 3 本は 1024x1024）。numpy では置き換えられないので、最初の Resize の直前
（`[1,32,256,256]`）で切った body を fp32 直接入力でコンパイルして内訳を測った
（コンパイル 1033 秒、metaDef 1 個・443 ノード）。

| 区間（256 タイル 1 回） | 時間 |
|---|---|
| 全体モデル（NPU） | 約 170 ms（853x480 の 12 タイルで 1 枚 2.07 秒） |
| body のみ（NPU） | 中央値 158.7 ms（最小 142.1 ms） |
| 高解像度の末尾のみ（CPU EP、fp32、全コア） | 約 120 ms |

NPU 上の時間の 9 割以上は body（RRDB 10 ブロック、Conv 156 本・Concat 120 本）で、
末尾が占めるのは 10〜15 ms 程度。末尾を CPU へ出して重ねても 1 タイル約 159 ms で約 7% しか縮まず、
CPU を全コア使う代償に見合わない。RRDB は全層が律速（1 層あたり約 1 ms。dense 接続で 32〜160ch の
256x256 活性を層ごとに出し入れする。spill が支配的かどうかは AI Analyzer では未確認）。

### 5.5 SwinIR-M（window attention）

英語の要約: [swinir-npu.md](../swinir-npu.md)

静止画の最高画質枠として SwinIR-M（`003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN`、real-world SR、
Apache-2.0）を追加した。`export_spandrel.py` で 256x256 固定 fp32 ONNX 化（torch/ORT 最大差 7.7e-07）。
attention 系（MatMul / Softmax / LayerNormalization / roll）をこの構成で NPU コンパイルした最初のモデル。

2026-08-29 時点の測定（853x480 / 1920x1080 → 4 倍）:

| 経路 | 精度 | 256 タイル 1 枚 | 480p 1 枚 | 1080p 1 枚 | 忠実度 |
|---|---|---|---|---|---|
| GPU（DirectML） | fp32 | 約 4.5 秒 | 約 60 秒（12 タイル） | 213 秒（45 タイル） | - |
| NPU（VitisAI 1.8.0） | bf16 | 6.6 秒 | 約 90 秒 | 推定 297 秒 | 38.5 dB |

**VAIML コンパイラの assertion と回避**

bf16cast モデルの VitisAI セッション生成が、ONNX → ONNX-MLIR lowering 段の native assertion
（`"Iteratees do not have equal length"`、exit 0x80000003）で 2.8 秒で停止した。
20 を超える切り分け（部分グラフ抽出と合成最小グラフ）の結果:

- 単体ではすべて成功: LayerNormalization / Softmax / 4D MatMul / roll 等価の Slice+Concat /
  window 分割の Transpose・Reshape 連鎖 / マスク Add 付き window attention / 非シフトの完全ブロック
- 失敗するのは「負の starts/ends リテラルを持つ Slice 2 本 + Concat」の組み合わせのみ
  （`torch.roll(x, shifts=+4)` の逆 roll の export 形）。最小再現器は 3 ノード。
  同一形状・同一構成で符号を非負にしただけの順 roll は成功する。
- スタックは `lowerONNXToONNXMLIR` → `ResultNamesUpdater` → `zip_equal<SmallVector<string>>`

負の Slice 境界リテラルを正の等価値へ書き換えるだけで回避できる（数値は bit-exact、ノード数・op 種は不変。
フルモデルで 72 箇所）。書き換え後は 1697 ノードの全体がコンパイルを通り、VAIML が 4971 op / 2969 GOPs を
100%・単一サブグラフで受理した（初回コンパイル 51 分、以後はキャッシュ）。書き換えは `export_spandrel.py` に
組み込み済み（`_rewrite_slice_nonneg`、export 時に常時適用）。
AMD へ報告済み（[amd/RyzenAI-SW#397](https://github.com/amd/RyzenAI-SW/issues/397)）。

**採用判断はリソース占有率で行った**

SwinIR-M は 256 タイル 1 枚で 2969 GOPs あり、NPU は GPU より遅い。採用理由は占有率
（2 秒間隔サンプリング、アイドル差分）:

| 実行中 | iGPU 3D エンジン | CPU 全体 | NPU | メモリコミット増 |
|---|---|---|---|---|
| NPU（VitisAI） | 2.4%（アイドル同等） | +0.7pt | 100%（時間積分） | +7.65GB |
| GPU（DirectML） | 99.1% | +0.8pt | 0% | +3.19GB |

NPU 実行中は iGPU が空いたままなので、他の作業と並行してバックグラウンドで実行できる。
TDR ライブダンプによる周期的なコミット増は 3.7。

**NPU 時間の内訳（2026-09-19）**

AI Analyzer（コンパイル 3820 秒、1 run 6.4〜7.4 秒、AIE レイヤ時間の合計 約 5.6 秒）。36 ブロック合計、2 run 平均:

| レイヤ（融合後の名前） | 個数 | 合計 | 割合 |
|---|---|---|---|
| `MatMul+MatMul_post_reshape+Gather+Mul`（attn/qkv。q/k/v の取り出し） | 108 | 2114 ms | 37.6% |
| `Transpose+MatMul+Add`（k の転置 → q·kᵀ → 相対位置バイアス加算） | 36 | 1270 ms | 22.6% |
| LayerNormalization | 128 | 247 ms | 4.4% |
| attn/proj（`MatMul+…+Transpose`） | 72 | 227 ms | 4.0% |
| `Softmax+MatMul` / attn の `MatMul` | 54 / 54 | 190 / 182 ms | 3.4% / 3.2% |
| MLP fc1（`MatMul+Div+Erf+Add+Mul`）/ fc2（`MatMul+Add`） | 36 / 42 | 165 / 181 ms | 2.9% / 3.2% |
| Conv 全部（浅い特徴抽出・各 RSTB 末尾・高解像度の末尾） | 21 | 約 130 ms | 2.4% |

- 時間の 6 割は 1 ブロックあたり 2 か所（qkv の 1 サブレイヤ 48〜65 ms、`Transpose+MatMul+Add` 約 35 ms）。
  行列積そのもの（MLP、proj、Softmax）は 1 層 3〜5 ms。
- 高解像度の末尾（Resize × 2 と Conv 4 本）は約 67 ms（1%）で、tail-cut の対象にならない。

**attention の書き換え実験（全体への適用は見送り）**

「qkv の 5 次元 reshape → permute → Gather と、k の転置・バイアス定数の展開が並べ替えコストになっている」
という推測を、layers.2 の blocks.0 → blocks.1 を切り出した 2 ブロックモデルで検証した
（bf16cast、コンパイル約 150 秒、warmup 2 + 10 run の中央値。各候補は CPU EP fp32 で元と一致を確認）。

| 候補 | 1 run | 備考 |
|---|---|---|
| 元のまま（2 回コンパイル） | 0.375 秒 / 0.406 秒 | 同一モデルの再コンパイルで 8% 変動 |
| V1: qkv を q/k/v の 3 本の MatMul に分割 | 0.332 秒 | fp32 で max abs diff 0.0 |
| V2: V1 + k を転置済みの向きで生成 | 0.334 秒 | V1 からの上積みなし |
| V3: V2 + `q*scale` を重みへ畳み込み（3 回コンパイル） | 0.262 / 0.323 / 0.328 秒 | 初回の 0.262 秒は再現せず |
| V4a: 相対位置バイアス加算を除去（切り分け専用） | 0.328 秒 | 重い層は消えない |

- 推測は否定された。切り出したモデルでは qkv の融合層は 3〜4 ms、`Transpose+MatMul+Add` は 3.5 ms。
  代わりに 30〜40 ms 級の層が `Add+Softmax`・`Softmax+MatMul`・`DataMovement` に現れる（2 ブロックで 4 か所・計 153 ms）。
  重い層は特定の演算に固有ではなく、`[1024,6,64,64]`（2516 万要素）のスコア系テンソルの出し入れを
  コンパイラがどの層に割り当てたかで決まる。
- 書き換えの実効果は約 15%（元 2 点の平均 0.390 秒 → 書き換え 5 点の平均 0.329 秒）。
  大きな中間テンソルが減り、30〜40 ms 級の層が 4 → 3 か所になる（AIE 計 280 → 254 ms）。
- 再コンパイルによる変動（±8%）と同程度の効果で、全体モデルの再コンパイル（約 1 時間）と
  キャッシュ・配布物の差し替えに見合わないため見送った。

**軽量版 SwinIR-S の検討（2026-09-05〜06、不採用）**

SwinIR-S（002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4、878K パラメータ、bicubic 劣化で学習）を
同じ手順で NPU 化した。VAIML は 3317 op / 272.9 GOPs を 100% 受理し、コンパイル 36.6 分。負の Slice 境界の書き換えは 0 件。

| 項目 | SwinIR-M | SwinIR-S |
|---|---|---|
| 演算量（256 タイル） | 2969 GOPs | 273 GOPs |
| NPU 1 タイル | 6.6 秒 | 2.98 秒 |
| GPU（DirectML）1 タイル | 約 4.5 秒 | 1.29 秒 |
| NPU bf16 と fp32 の一致 | 38.5 dB | 47.7 dB |

演算量が 1/11 でも NPU は 2.2 倍速にしかならない（実効 92 GOPS）。
チャネル幅の削減は効かず、層数（op 数）に比例したコスト（約 0.9 ms/op）が残る。

画質は目視で不採用。bicubic / Real-ESRGAN x4plus / SwinIR-M / SwinIR-S の 4 列比較で、SwinIR-S は
小さな正方形のタイルを敷き詰めたような格子状の破綻が全体に出て、4 者の中で最も悪かった。
bicubic 劣化で学習した軽量モデルは実写・アニメの実素材には向かない。GUI には登録しない。

### 5.6 AdcSR（1 ステップ拡散）

英語の要約: [adcsr-npu.md](../adcsr-npu.md)。タイル継ぎ目の格子の診断は [adcsr-tile-diagnosis.md](https://github.com/tomhda/ultraeasy-upscaler/blob/main/docs/adcsr-tile-diagnosis.md)。

AdcSR（net_params_200、SD2.1-base 派生・1 ステップ）は、1 グラフで NPU に常駐させると 2 回目以降の出力が
全画素 NaN になる。切り分けの経緯:

1. 1 グラフで 2 回目以降 NaN。73 サブグラフに分割された状態では正常。
2. GroupNorm 由来の 3 次元正規化を 4 次元に書き換えて 1 サブグラフ化しても NaN。
3. UNet 出力の直後で前半 F / 後半 G に切断すると、F 単独・G 単独は何回でも正常。
4. 同一プロセスで G を 1 回実行すると、F が以後ずっと NaN になる（G → F の一方向）。
5. 別プロセスに分けると 100/100 回正常。

条件: 正規化書き換え済みの bf16cast、境界 3 テンソル（main `[1,256,64,64]`、mean/std `[1,3,1,1]`、
いずれも float32。名前で照合）。F の VAIML 受理は 4786/4797 op・498.809 GOPs（単一サブグラフ、約 93 分）、
G は 1520/1520 op・658.343 GOPs（約 30 分）。
逐次実行で 2.05 秒/タイル（F 0.71 + 転送 2.5 ms + G 1.33）。F と G を重ねて実行すると 2.20 秒/タイルと
遅くなるため逐次に固定した。

実装は `tools/npu-serve/npu_worker.py`（役割別のテンソル実行器）と `tools/npu-serve/npu_twostage.py`
（起動時セルフテスト、タイル単位の健全性検査、1 画像 1 回の内部復旧、復旧不能時は画像単位で
同じ AdcSR の DirectML へ再処理）。`UEU_ADCSR_NPU2=0` で GPU 実行に戻る。
AMD へ報告済み（[amd/RyzenAI-SW#402](https://github.com/amd/RyzenAI-SW/issues/402)）。

### 5.7 SinSR の検討（2026-09-12、不採用）

AdcSR と同じ「1 ステップの拡散系超解像」である SinSR（ResShift の蒸留、CVPR 2024）が AdcSR より速いかを、
NPU に載せる前に DirectML（Radeon 860M）で確かめた。公式コードの推論経路
（LR 前処理 → VQGAN encode → 1 ステップ UNet → VQGAN decode）を 1 モジュールに包んで固定形状 ONNX
（64→256 と 128→512、opset 17）にし、公式推論と一致を確認した。

| | AdcSR 128→512 | SinSR 64→256 | SinSR 128→512 |
|---|---|---|---|
| ONNX ノード数 | 1510 | 1717 | 1729 |
| GOPs/タイル | 1099 | 1111 | 5267 |
| 出力 1MP あたり GOPs | 4193 | 16948 | 20093 |
| DirectML 1 タイル | 1.1〜1.6 秒 | 0.73 秒 | GPU ハング（計測不可） |
| tos 1280x534 1 枚 | 約 100 秒（84 タイル） | 502 秒（680 タイル） | — |
| ライセンス | Apache-2.0 / OpenRAIL-M | CC BY-NC-SA 4.0 / S-Lab 1.0（非商用） | 同左 |

- 出力画素あたりでは AdcSR の約 2.2 倍遅い。主因は VQGAN の encode/decode が出力サイズ（512 角）で動くことと
  Swin attention（MatMul 40・Softmax 20・Erf 18）。
- 128→512 版は VQ の距離計算 `Einsum`（16384×8192、約 537MB の中間テンソル）が iGPU のメモリを超えて
  DXGI 887A0006 でハングする（64 版の同テンソルは約 134MB で完走。サイズ依存と整合）。
- 画質は tos（実写）で AdcSR が明確に上（SinSR は平滑化が強い）。bbb（アニメ）は背景に生成ノイズのまだらが出る。
- NPU に載せた場合の見込みは op 数換算で約 6.5 秒/タイル（約 0.9 ms/op から。推測）で、AdcSR の約 3 倍。

DirectML の ONNX Runtime は、shape に `-1` を含む Reshape でセッション初期化に失敗する
（`MLOperatorAuthorImpl.cpp(2879)`、E_INVALIDARG。1 ノードの最小プローブで再現）。
shape inference で求めた具体値に書き換えると通る（SinSR 128 版で 196 箇所）。
同じ shape 定数を形状の異なる Reshape が共有している場合は、形状ごとに定数を複製する必要がある。
Erf・Einsum・ArgMin・cubic/nearest Resize・3D InstanceNormalization・Softmax（4D/5D）・
負境界の Slice+Concat は DirectML では問題なかった。

## 6. 実行系の経緯

### 6.1 2026-08 上旬: PNG 入出力の Python ワーカー（旧構成）

実効値は 854x480 → 4 倍・12 フレームのフォルダ一括の平均。GPU は realesrgan-ncnn-vulkan（fp16）。

| 構成 | 精度 | タイル速度 | 実効/枚 |
|---|---|---|---|
| GPU + Anime Video v3 | fp16 | — | **0.7 秒** |
| NPU + Anime Video v3 | bf16（分解） | **151 ms** | 約 3.1 秒 |
| NPU + Real-ESRGAN | bf16 | 168 ms | 約 3.7 秒 |
| NPU + Real-ESRGAN Anime | bf16 | 262 ms | 約 4.5 秒 |
| GPU + General Video v3（強/弱） | fp16 | — | 参考: 720p 単発 約 3 秒 |
| GPU + Real-ESRGAN | fp16 | — | 参考: 単発 17 秒 |

NPU の実効はタイル推論の理論値（1.8〜3.1 秒/枚）に対し、Python ワーカーのフレーム入出力
（PNG 読み書き・タイル分割結合）が約 1.2〜1.4 秒/枚を上乗せしていた。
この時点では、NPU の利点は速度ではなく GPU をほぼ占有しない点だった。

当時の目視評価:

- アニメ/CG: GPU + Anime Video v3 が最良。NPU + Real-ESRGAN Anime（bf16）との差は小さく、好みの範囲。
- 実写: Anime Video v3 系は平滑化が強く、肌の質感が失われる。Real-ESRGAN 系が上。
- General Video v3（ノイズ除去弱、wdn）は原本の質感を残す。Real-ESRGAN は輪郭を強調する。

当時の用途別の構成:

| 用途 | 構成 |
|---|---|
| 実写・汎用の動画 | NPU + Real-ESRGAN（bf16） |
| アニメ（画質優先） | NPU + Real-ESRGAN Anime（bf16） |
| アニメ（速度優先） | GPU + Anime Video v3 |
| 実写（質感重視） | GPU + General Video v3（ノイズ除去弱） |
| 静止画（最高画質） | GPU + SwinIR-M |
| 静止画（最高画質・GPU をほぼ占有しない） | NPU + SwinIR-M（bf16） |

現在のモデル選択の目安は README の「モデル選択の目安」。

比較画像（列は左から: オリジナル（bicubic）/ GPU + Anime Video v3 / NPU + Anime Video v3 /
NPU + Real-ESRGAN / GPU + Real-ESRGAN）:

![Big Buck Bunny](../images/quality_matrix_bigbuckbunny.png)

![Superman 1941](../images/quality_matrix_superman1941.png)

![Tears of Steel](../images/quality_matrix_tearsofsteel.png)

### 6.2 2026-08-14〜15: Windows ML 経由の実行と常駐 serve モード

Ryzen AI SW を直接使う代わりに、OS 標準の Windows ML（ONNX Runtime + EP カタログ）経由で同じモデルを
実行する検証。実装は `tools/winml-sr`（C# コンソール、MSIX 不要・unpackaged 動作）。
fp32 ONNX を渡すだけで、量子化・bf16 変換・キャッシュ管理は EP 側が自動で行う。

実測（854x480 → 4 倍、PSNR は同モデルの CPU fp32 比）:

| 構成 | ms/タイル | 1 枚（推論） | PSNR |
|---|---|---|---|
| Anime Video v3 × DirectML（GPU） | 52 ms/256 | **0.63 秒** | 101 dB |
| Anime Video v3（分解版）× VitisAI（NPU）512 タイル | 512 ms/512 | **1.02 秒** | 51.6 dB |
| Real-ESRGAN（RRDB）× VitisAI（NPU） | 170 ms/256 | 2.06 秒 | 47.2 dB |
| Real-ESRGAN（RRDB）× DirectML（GPU） | 231 ms/256 | 2.78 秒 | 94.9 dB |

- DirectML は同じ 860M を使う realesrgan-ncnn-vulkan より約 2 倍速い。
  MIGraphX（69 ms）/ RyzenAILight（227 ms）は DirectML（52 ms）に劣り不採用。
- EP は Microsoft Store ではなく Windows Update 経由で配信されるため、WU の一時停止中はダウンロードできない
  （公式の既知事項）。取得後は停止中も動作する。
- 初回の VAIML コンパイル（1〜9 分）は EP が自動で永続キャッシュし、以後のセッション生成は 0.1〜1.2 秒。
- 常駐 serve モード（stdin/stdout の生ピクセルパイプ）で、フレーム入出力のオーバーヘッドを
  1.2〜1.4 秒 → 約 50〜60 ms/枚 に短縮した。

### 6.3 2026-08-15: Ryzen AI SW 1.8.0 への更新

Windows ML の EP カタログが停止した（NPU ドライバ .329 が VitisAI EP 1.8.68 の対応上限 .297 を超過。
詳細は [span-bench-results.md](https://github.com/tomhda/ultraeasy-upscaler/blob/main/docs/span-bench-results.md)）ため、ネイティブ側を 1.8.0 に更新した
（conda env `ryzen-ai-1.8.0`、1.7.1 と並存、キャッシュは `vendor/amd-npu-1.8/` に分離。
1.8.0 はドライバ .280 以降に対応、推奨 .376）。

512 タイル bf16、853x480 → 4 倍、タイル処理込みの実測:

| モデル | 1.7.1 | 1.8.0 | 画質（fp32 比） |
|---|---|---|---|
| Anime Video v3（分解版） | 1.305 秒/枚 | **1.139 秒/枚** | 48.3 dB（両版同値） |
| 4xNomosUni SPAN | 0.707 秒/枚 | **0.602 秒/枚** | 43.1 dB（両版同値） |

- コンパイラの更新で一律 13〜15% 高速化した。
- VAIML コンパイル時間は増えた（Anime Video v3 分解版 512: 389 → 914 秒）。初回のみ。

以後の高速化（tail-cut、CPU 側の削減）は 5.2・5.3。

## 7. 再現手順（`scripts/npu/`）

```text
export_animevideov3.py   SRVGGNetCompact → 固定 256x256 fp32 ONNX（--tile で 512 可・--size WxH で非正方形可）
export_x4plus_anime.py   RRDBNet(6B) → 同上
export_spandrel.py       spandrel 対応モデル（SPAN / SwinIR 等）→ 固定サイズ fp32 ONNX（負の Slice 境界を書き換え）
split_tail.py            末尾 DepthToSpace 以降の切断（body と tail マニフェストの生成、CPU 一致検証）
make_calib_patches.py    フレーム → 256px キャリブレーションパッチ（int8 用）
quantize_animevideov3.py Quark XINT8 量子化（int8 用）
verify_animevideov3_npu.py  NPU コンパイル・割当・PSNR・速度の一括検証
```

bf16 の手順:

```text
python -m quark.onnx.tools.convert_fp32_to_bf16 --input <fp32.onnx> --output <bf16.onnx> --format with_cast
初回セッション生成でキャッシュディレクトリに modelcachekey_<stem>/ が作られる（数分〜約 1 時間）
```

- キャッシュキーは ONNX ファイル名の stem から決まる。モデルファイルと `modelcachekey_<stem>/` は対で扱う。
- VAIML の初回コンパイル時間の目安: Anime Video v3 / SPAN 約 9 分、Real-ESRGAN（RRDB）約 19 分、
  SwinIR-M 約 51 分、AdcSR 前半約 93 分 + 後半約 30 分。
- コンパイル中に EP がカレントディレクトリへ書く `original-model-signature.txt` /
  `original-info-signature.txt` を消すと、コンパイル末尾のコピーで失敗する。
