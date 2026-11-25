# Data

## 1. Public sources

| Dataset | What we use | Obtained by | License |
|---|---|---|---|
| MAEC (Li, Yang, Smyth & Dong, CIKM 2020) | transcripts (`text.txt`, one line per sentence) and the audio clip of every line, 2015 and 2016 calls | notebooks 1 and 2 download the sentence-level audio archive (`sp1500_earningscall.zip`) | CC BY-SA 4.0 (stated in the dataset's repository) |
| EC (Qin & Yang, ACL 2019) | transcripts (`TextSequence.txt`) and line-level audio (`CEO/`), 2017 calls | notebooks 1 and 2 download the ACL19 release from the dataset's Google Drive folder | none stated; released by its authors for reproducing their results |
| KeFVP volatility files (Niu et al., Findings of EMNLP 2023), commit `95993892d087599dc5530804aada73f7b5004f39` | train/dev/test splits, targets `future_3/7/15/30`, MAEC historical features `past_1 … past_29`, the EC test file | downloaded from `github.com/hankniu01/KeFVP` (`price_data/`) | none stated |

**Targets.** The realized log-volatilities are KeFVP's released targets, used unchanged (paper, Section III).

**Splits.** KeFVP's chronological 70:10:20 splits. MAEC-15: 535 / 76 / 154 calls; MAEC-16: 980 / 140 / 280. The EC models are trained on the MAEC-15 and MAEC-16 training splits, selected on their development splits, and tested on the EC test calls.

## 2. Files in this repository

### `data/ec/`

| File | Content |
|---|---|
| `ec_test_corrected.csv` | The EC test calls (110 of 112), with `ticker, name, year, month, day, text_file_name`, the released targets `future_3, future_7, future_15, future_30`, and the 29 historical-volatility features `past_1 … past_29` recomputed with the MAEC definition. Notebook 3 reads this file. |
| `ec_adjusted_closes.csv` | Daily adjusted closing prices (one column per ticker, 2017-08-28 to 2017-12-21) for the EC test calls. An excerpt of a price table compiled by the authors (1,151 tickers, 2015 to 2018); its data provider was not recorded. |
| `ec_adjusted_closes_supplement.csv` | Yahoo Finance adjusted closes (`Adj Close`, retrieved with `yfinance` 1.8.0 on 2026-11-04) for six calls the first file cannot serve. The column names are the EC file's tickers; the source symbols are ABC→COR, BLL→BALL, cbre→CBRE, DG→DG, PKI→RVTY, FB→META. |

`scripts/build_ec_features.py` rebuilds `ec_test_corrected.csv` byte for byte from the two price files and KeFVP's EC test file. The definition, for a call on trading day $d$, with daily returns $r_t = p_t/p_{t-1}-1$:

$$v_j = \ln\sqrt{\frac{1}{30-j}\sum_{t=d-29}^{d-j}(r_t-m)^2},\qquad m=\frac{1}{30}\sum_{t=d-29}^{d} r_t,\qquad j=1,\dots,29.$$

Adjusted prices depend on the day they are retrieved, because later dividends and splits change them. Prices downloaded today therefore do not reproduce the two price files exactly; the files are kept so that `ec_test_corrected.csv` can be rebuilt from the same inputs.

Ticker matching: the EC ticker as written, else upper case, else the alias RHI.F → RHT (Red Hat). CBS and WRK (both 2017-11-02) have no usable price history and are excluded.

### `results/predictions/`

Every test prediction behind our results in Tables I and III, one gzip CSV per dataset, in the format notebook 3 writes (`item_losses.csv`):

| Column | Meaning |
|---|---|
| `seed` | one of the 55 seeds |
| `dataset` | `maec15`, `maec16` or `ec` |
| `modality` | QA representation: `text_audio_prompt2` (Volatility Prompt), `text_audio_prompt1` (General Prompt), `text_only`, `audio_only`, `chunk_text_audio`; `vol_only` for Past Volatility |
| `fusion_type` | `residual_gated` (the two-stage model), `concat`, `qa_only`; `n/a` for Past Volatility |
| `ticker`, `date` | the test call |
| `horizon` | 3, 7, 15 or 30 |
| `pred`, `target`, `sq_err` | prediction, target, squared error |

| File | Configurations | Test calls with a prediction |
|---|---|---|
| `maec15_predictions.csv.gz` | Past Volatility; Proposed | 154; 150 |
| `maec16_predictions.csv.gz` | the eight rows of Table III | 280 (Past Volatility, Transcript Chunks + Audio); 272 (QA-based rows) |
| `ec_predictions.csv.gz` | Past Volatility; Proposed | 110; 109 |

A QA-based model has no prediction for a call without QA units. `scripts/make_tables.py` scores every call, giving such a call the same seed's Past Volatility prediction.

## 3. Data package (Google Drive)

**Link: <https://drive.google.com/drive/folders/1LOMRqqFQrA-Oz4n6GBDxlLWOfPE8VxZp>**

On Colab, put the folder in your Google Drive as `My Drive/qa-grounded-volatility/`, so that the notebooks find `qa/` and `embeddings/` there; on your own machine, put `qa/` and `embeddings/` in the work folder (`WORK_DIR` in the settings cell; by default `notebooks/qa-grounded-volatility-work/`). Notebook 3 can also download and verify `embeddings/` itself. Check the downloaded files against the SHA-256 values listed below.

### `qa/`: QA units (output of notebook 1)

| File | Bytes | SHA-256 |
|---|---|---|
| `kefvp_train_maec15_chunks_qas_with_final_label_with_textlines_with_audiomap.json` | 11,010,131 | `ea866d035ef73a4130ff65e0fc5afa7dad5e198ca55dffc4bd1f42f7ce13d121` |
| `kefvp_dev_maec15_chunks_qas_with_final_label_with_textlines_with_audiomap.json` | 1,614,182 | `f41a12f1d69105eaaf6f8542dbec64f8358adbd102cccd48e29e355f773f0f36` |
| `kefvp_test_maec15_chunks_qas_with_final_label_with_textlines_with_audiomap.json` | 3,087,714 | `91e7e5e601c71ecd52cd03e111e08c06fd90bcb23bbdc31996db9f0a8b802ea3` |
| `kefvp_train_maec16_chunks_qas_with_final_label_with_textlines_with_audiomap.json` | 21,746,777 | `576fdcb86acfcc53931e8ee45d7eb4a8e7ec6f333de22e4b2ba712b2208bbb74` |
| `kefvp_dev_maec16_chunks_qas_with_final_label_with_textlines_with_audiomap.json` | 3,166,602 | `8ae24e02fbeb7b6270002f80f3d9ac804add1bf5fa9e8d287b3c48412f75bc7d` |
| `kefvp_test_maec16_chunks_qas_with_final_label_with_textlines_with_audiomap.json` | 5,097,703 | `f0d0817a15e1d10ccd63da00177b6964be99f29272ba2ca9a19195eb9544f8da` |
| `kefvp_test_ec_chunks_qas_with_final_label_with_textlines_with_audiomap.json` | 3,869,921 | `9c17b062edbb9601cc39659f3b26c7f60a09528e25f7a1654481831db3e9f5c2` |
| `call_map.json` | 343,533 | `76c9a9081c4d3b132ec2ba3fb786c0a6cfd733a73b5241b8ab3d62a3f0939f4c` |
| `kefvp_train_maec_35clusters.json` | 158,598 | `3fc31dbb487894ba558118396d7e45fed41b560774eb6bea54fed6bb81160004` |

**QA files.** A JSON object keyed by `"<call id>-<chunk index>"`:

```json
"0-0": {
  "qas": [
    {"Q": "How might changes in operating margin expectations due to planned spending affect near-term profitability?",
     "A": "Management indicated they are guiding to approximately a 20% operating margin, ...",
     "source_lines": [81, 83],
     "topic": "Mix and operating leverage margin volatility"}
  ],
  "text_lines": ["[TURN 1] Thank you, Hikmet", "[TURN 1] Third quarter reported revenues of $1.4 billion ...", "..."],
  "audio_map": {"1": "<speaker>_1_1.mp3", "2": "<speaker>_1_2.mp3", "...": "..."},
  "audio_path": "/content/original_dataset/ACL19_Release/<call folder>/CEO"
}
```

- `source_lines` is an inclusive span `[first, last]` of 1-based line numbers within the chunk's `text_lines`. Notebook 2 reads the first two values as that span.
- A few records deviate from this: 40 of the 19,734 QA units list more than two values (only the first two are used), and 9 have an unusable span (8 whose first value is larger than the second, one of them among the 40; 1 running past the chunk), so they have no embedding in the variants that use the cited lines.
- `topic` is one of the 35 topics, except for two QA units whose label is not in the list (`Earnings guidance visibility risk`, a lower-case variant, in the MAEC-15 training file; `Revenue mix shift volatility` in the EC file).
- `audio_map` gives the audio clip of each line.
- `audio_path` is the clip folder after notebook 2's download on Colab (under `/content`; elsewhere notebook 2 points it to its `AUDIO_DIR` in memory), so the audio of a QA unit is the concatenation of the clips of its source lines.

The EC file holds 111 of the 112 EC test calls: it includes CBS and WRK, which are not evaluated (no usable price history), and lacks American Tower (2017-10-31), which has no transcript in the EC release.

**`call_map.json`.** `{"maec": {...}, "ec": {...}}`, mapping a call id to `{"ticker", "date" (YYYYMMDD), "text_file_name"}`.

**`kefvp_train_maec_35clusters.json`.** The 35 topics: for each, its name (`cluster_label`) and the preliminary labels it merges (`member_labels`).

### `embeddings/`: QA embeddings (output of notebook 2)

19 pickle files, `kefvp_<split>_<dataset>_<variant>.pkl`: the files the paper's models use. The Volatility Prompt variant is given for the splits `train`, `dev`, `test` of `maec15` and `maec16` and `test` of `ec` (7 files); the four ablation variants are given for the three `maec16` splits only (12 files), because the ablation of Table III is on MAEC-16. Notebook 2 writes all five variants for every split (35 files); notebook 3 reads only these 19.

| Variant suffix | Paper |
|---|---|
| `_qa_text_audio_prompt2_embeddings` | QA Text + Audio, Volatility Prompt (proposed model) |
| `_embeddings` | QA Text + Audio, General Prompt |
| `_text_embeddings` | QA Text |
| `_audio_embeddings` | QA Audio |
| `_chunk_text_audio_embeddings` | Transcript Chunks + Audio |

Each file is a list of records:

```python
{"ecc_chunk_id": "0-0", "qa_index": 0, "question": "...", "topic": "...",
 "embedding": np.ndarray(shape=(3584,), dtype=float32),
 "ticker": "WU", "date": "20171102", "call_id": "Western Union Co_20171102"}
```

Chunk-level records have no `qa_index`, `question` or `topic`. The SHA-256 of each file (notebook 3 checks the files against these values):

| File | SHA-256 |
|---|---|
| `kefvp_dev_maec15_qa_text_audio_prompt2_embeddings.pkl` | `df70402a856e90fdf3919b79a128256a4f908428e63c136f029ecff6630770d4` |
| `kefvp_dev_maec16_audio_embeddings.pkl` | `3cb99daf22f375a2384007733e4f90b9b8d43073f234a42794a073b7ef33ca77` |
| `kefvp_dev_maec16_chunk_text_audio_embeddings.pkl` | `1850ff3dbb15202af5902b09bcffc0ef652e6267896905b0fcfa56fb5d5c45c6` |
| `kefvp_dev_maec16_embeddings.pkl` | `82bc523bc456fabbe9f1c68cf98b6dedca2a521f03479d858397e874ee131f3e` |
| `kefvp_dev_maec16_qa_text_audio_prompt2_embeddings.pkl` | `21c702b926af72e6fcbb3993898d9c4c0e013153ec439c210ecb824338665b95` |
| `kefvp_dev_maec16_text_embeddings.pkl` | `495d9b7a7f14b87be45911488348c8720553f401e72c7715a5661bdbbf2a3c4a` |
| `kefvp_test_ec_qa_text_audio_prompt2_embeddings.pkl` | `226e28c2ea362c66a60f6d0d6337b87a7ffbf124a00941a613376ce1cb54de10` |
| `kefvp_test_maec15_qa_text_audio_prompt2_embeddings.pkl` | `cb37cb0a12ad136882c3478cc37c2c65a4e8b5d9791425833109684f00f9f526` |
| `kefvp_test_maec16_audio_embeddings.pkl` | `3f5d0d51116d31cbc410ec7c85dde609c07e9239dc4482478f06393b8777ef1d` |
| `kefvp_test_maec16_chunk_text_audio_embeddings.pkl` | `cb7a4d8cf90d0a49545d632aa82cc97a0f2d339ee06fcaa907daa4d46e983308` |
| `kefvp_test_maec16_embeddings.pkl` | `da9e369e7c9b0506c01a38d558bf683392a5f58bdab6c8868ebc96f8b3a71eff` |
| `kefvp_test_maec16_qa_text_audio_prompt2_embeddings.pkl` | `23af33389762c58fa10ea3808a6fb8e65c59f6e2c78c93d1bfd0638d35e49d3e` |
| `kefvp_test_maec16_text_embeddings.pkl` | `e2ce2570b5fdf070ef5fbddce34b109154951ed20faba5dcdec00b4f1110624f` |
| `kefvp_train_maec15_qa_text_audio_prompt2_embeddings.pkl` | `04056abd9bef62eb22b3031631bd08acaa08c26ba34d5381c1b364dd4c6af9fb` |
| `kefvp_train_maec16_audio_embeddings.pkl` | `b3918575c77a2b1f0a15592763479a594a65019cd5c1dc867a7cea1203e1a55f` |
| `kefvp_train_maec16_chunk_text_audio_embeddings.pkl` | `c10201c93b59e3604eb0eeda77cb7c51d15657639ba6149d643e772b3de69ff6` |
| `kefvp_train_maec16_embeddings.pkl` | `b2f64b2cfd00d1fd74b2bdbd7ef52436ab250c31b979b59b97705ce5f87b846b` |
| `kefvp_train_maec16_qa_text_audio_prompt2_embeddings.pkl` | `892aaabf9a182647ba0d22f16244303f8079a50bb7cf3b061d8ab0f34cabadb0` |
| `kefvp_train_maec16_text_embeddings.pkl` | `5940eab89f718527619b9397349b4fc651b7664009c8b8a7eedc0b8f2f6e7073` |

## 4. Terms of use

The MIT license applies to the original source code. Third-party data remain subject to their respective terms.

The official MAEC dataset (Li et al., CIKM 2020) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The QA files and embeddings incorporate or are derived from MAEC and EC transcripts and audio. Applicable upstream licensing conditions, including ShareAlike requirements, apply.

The volatility targets and dataset splits are obtained from KeFVP (Niu et al., Findings of EMNLP 2023). The price files support the recomputation of the EC historical-volatility features.
