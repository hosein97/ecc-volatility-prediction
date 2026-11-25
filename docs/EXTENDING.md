# Running your own experiments

This page shows where to add things without editing the paper's code (sections 1 to 6), how to run the notebooks on your own machine (section 7), and what to do when a dataset download is refused (section 8). In the notebooks, the cells labelled `[EXPERIMENT]` are the paper's code; add your changes in **new cells**, and write to your own output folder, so that you can always go back to the paper's run.

Before you start, run the notebook once in `"smoke"` mode to see that your setup works ([README](../README.md#reproducing-the-paper)).

## 1. Analyse the released predictions (no GPU)

Every test prediction of the paper is in `results/predictions/` ([format](DATA.md)). One row is one seed, test call and horizon. Run this from the repository's top folder:

```python
import pandas as pd

df = pd.concat(pd.read_csv(f"results/predictions/{d}_predictions.csv.gz", keep_default_na=False)
               for d in ("maec15", "maec16", "ec"))

def avg_mse(df):
    """Average MSE per model: MSE per horizon, mean over the four horizons, then mean and spread over seeds."""
    per_seed = (df.groupby(["dataset", "modality", "fusion_type", "seed", "horizon"])["sq_err"].mean()
                  .groupby(level=[0, 1, 2, 3]).mean())
    return per_seed.groupby(level=[0, 1, 2]).agg(["mean", "std", "count"])

print(avg_mse(df).round(4))
```

`keep_default_na=False` matters: the Past Volatility rows have `fusion_type` `n/a`, which pandas would otherwise read as missing.

This function scores each model on the calls it has a prediction for. A QA-based model has no prediction for a call without QA units; the paper's tables give such a call the same seed's Past Volatility prediction (`scripts/make_tables.py` does this), so its numbers can differ slightly from this function's for the QA-based models.

The same function works on the `item_losses.csv` of your own run of notebook 3.

## 2. Add an experiment to notebook 3

An experiment is a triple `(dataset, QA representation, model)`. The paper's list is `EXPERIMENTS`, defined in the cell that starts with `# The configurations reported in the paper`. To add one, insert a **new cell after that cell and its `[SETUP]` cell, before the training cell**:

```python
EXPERIMENTS = EXPERIMENTS + [("maec16", "text_only", "concat")]      # (dataset, QA representation, model)
experiment_config_df = pd.DataFrame(EXPERIMENTS, columns=["dataset", "modality", "fusion_type"])
```

- **Datasets:** `maec15`, `maec16`, `ec` (EC is test-only: the model is trained on MAEC-15 + MAEC-16).
- **QA representations:** the keys of `MODALITY_SUFFIX`: `text_audio_prompt2` (Volatility Prompt), `text_audio_prompt1` (General Prompt), `text_only`, `audio_only`, `chunk_text_audio`.
- **Models:** the keys of `FUSION_MODEL_CLASSES`: `residual_gated` (the paper's two-stage model), `concat`, `qa_only`.

With this example the notebook trains 13 models instead of 12; the table step reports the paper's rows and ignores the others.

Things to know:

- **Embeddings.** The data package holds the Volatility Prompt embeddings for all datasets and the four other representations for MAEC-16 only (19 files). Another combination, for example `("maec15", "text_only", ...)`, needs its embedding files: compute them with notebook 2 (it writes all five representations for every split), put them in `EMBEDDING_DIR`, and load them in your new cell, for each split: `EMBEDDINGS[("maec15", "train", "text_only")] = load_embeddings("maec15", "train", "text_only")`.
- **Results.** `item_losses.csv` in the results folder is the complete record (every seed, call and horizon of every model). The notebook also writes one `results_*.csv` file per model, with the per-seed losses.
- **Output folder.** Set `OUTPUT_DIR` in the settings cell, so that your run does not replace the results of a run of the paper's list.
- **Fewer seeds.** Train a subset with a new cell after the first `[EXPERIMENT]` cell: `SEEDS = SEEDS[:5]`. The paper's results use all 55.

## 3. Add a model to notebook 3

Define your model class in a new cell after the paper's model cells, and register it:

```python
class MyModel(nn.Module):
    def __init__(self):                      # no arguments
        ...
    def forward(self, vol, qa, mask):        # same inputs as VolECCConcatModel; returns the four predictions
        ...

FUSION_MODEL_CLASSES["my_model"] = MyModel
```

Then add experiments with `"my_model"` as in section 2. A model registered under a new name is built without arguments (`model_cls()`) and called as `model(vol, qa, mask)`, like `VolECCConcatModel`. Only the name `residual_gated` is treated differently by the training code: that model is built from the pretrained Past Volatility model of the same seed (`model_cls(vol_model)`) and returns three values. A new model that builds on the frozen Past Volatility model therefore needs changes in the training cell, in a copy of it. `VolECCConcatModel` and `VolECCResidualGatedModel` are the templates to follow. This extension is described as a guide and has not been verified.

## 4. Add a QA representation (notebook 2)

Each section of notebook 2 is one representation: a prompt, the inputs it passes to Qwen2.5-Omni, and an output file suffix. To add one, copy the MAEC cell and the EC cell of the section closest to what you want into new cells, change the prompt in `messages` and the suffix in `save_path` (for example `_my_prompt_embeddings.pkl`), and run them, then section F, which attaches ticker and date to every file. In notebook 3, add the suffix in a new cell: `MODALITY_SUFFIX["my_prompt"] = "my_prompt_embeddings"`, load the files as in section 2, and add experiments with `"my_prompt"`. This extension is described as a guide and has not been verified.

## 5. New QA units or topics (notebook 1)

The prompts for QA generation and for the topics are in notebook 1, in the cells before each paid cell. A change there means a new paid run. Use `"mini"` mode first: it runs the whole pipeline on a small number of calls, for a small fraction of the full cost (not measured). Notebook 1 writes to `qa_generated/` (never to the released `qa/`); point notebook 2's `QA_DIR` to it.

## 6. Comparing with the paper

- Use the same seeds, splits and evaluation as the paper: the released predictions hold, for every seed, the prediction of every test call, so a new model can be compared with the paper's models seed by seed and call by call (join on `seed`, `dataset`, `ticker`, `date`, `horizon`).
- `scripts/sensitivity_and_twosided.py` and `scripts/confirmatory_bootstrap.py` show the paper's tests (seed-paired t-tests, call-level bootstrap with Holm correction) and can be adapted to a new model.
- Differences between complete training runs can be of the same order as the smallest effects in Table III; compare distributions over seeds, not single seeds.

## 7. Running the notebooks on your own machine

The notebooks are designed for Google Colab, where the paper's runs were made ([README](../README.md#where-the-notebooks-run)). These are the steps for running them on your own machine.

```bash
git clone https://github.com/hosein97/ecc-volatility-prediction
cd ecc-volatility-prediction
# with a GPU: first install PyTorch for it (https://pytorch.org/get-started/locally/); without one, skip this (the next line installs the CPU version)
pip install -r requirements-training.txt      # notebook 3  (requirements-qa.txt: notebook 1)
cd notebooks
jupyter notebook 3_volatility_prediction.ipynb
```

Edit the settings cell and run all cells, starting with `RUN_MODE = "smoke"`. By default the data and the results are read from and written to `notebooks/qa-grounded-volatility-work/` (set `WORK_DIR` for another folder); no Google Drive is involved. A few small downloaded files (the KeFVP volatility files) and, for a command-line run, the executed copy of the notebook are written into `notebooks/` itself. For notebook 3, `DOWNLOAD_PUBLIC_DATA = True` fetches and verifies the embeddings.

On Windows, keep the paths short: clone the repository into a short folder (for example `D:/qav`), and use a short folder for the audio of notebook 1 (`AUDIO_DIR`, for example `D:/qav_audio`). Windows limits a file path to about 260 characters; the installation of Jupyter and some audio files have long names.

**Without opening the notebook.** Give the settings as environment variables and execute the notebook:

```bash
export QAV_RUN_MODE=smoke                      # then check, then full
# for check mode, optional: export QAV_CHECK_DATASETS=maec15,maec16,ec   and   QAV_CHECK_SEEDS=1,3,5
export QAV_WORK_DIR=/path/to/your/work/folder  # holds embeddings/, receives results/
# optional: export QAV_DOWNLOAD_PUBLIC_DATA=1   to download and verify the embeddings into that folder
cd notebooks
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 3_volatility_prediction.ipynb --output run_3.ipynb
```

The `export` lines are written for a bash shell. On Windows, set the variables with `$env:QAV_RUN_MODE = "smoke"` (and so on) in PowerShell, or with `set QAV_RUN_MODE=smoke` in the command prompt, and run the same `jupyter` line. The command prints little: the notebook's output, including `SMOKE TEST PASSED`, is in the executed copy `run_3.ipynb`, which you can open in Jupyter.

**Notebook 1** is executed the same way (`pip install -r requirements-qa.txt`, then `1_qa_generation.ipynb --output run_1.ipynb`), with `OPENAI_API_KEY`, `QAV_RUN_MODE`, `QAV_WORK_DIR`, `QAV_QA_DIR` and `QAV_AUDIO_DIR`, and for the mini mode optionally `QAV_MINI_TRAIN_CALLS`, `QAV_MINI_EVAL_CALLS` and `QAV_MINI_EC_CALLS`. A variable that is set wins over the settings cell, so unset `QAV_RUN_MODE` when you no longer want it. In `smoke` mode the command ends with the error `SmokeTestPassed` and a non-zero exit code, and writes no output notebook: that is how the notebook stops before the paid cells, and `SMOKE TEST PASSED` is printed in the command's output. Unpacking the archives needs 7-Zip on Windows. The QA files store the audio folders under `AUDIO_DIR`, so QA files written on Windows cannot be used on Colab, or the other way round, without the audio at the same path.

## 8. Dataset downloads on Colab

The MAEC and EC archives are third-party files on Google Drive, which limits downloads of popular files, for up to 24 hours. Try again later. A way around a refused MAEC download on Colab: open the file in Google Drive (`https://drive.google.com/file/d/1m1GRCHgKn9Vz9IFMC_SpCog6uP3-gFgY/view`), choose "Add shortcut to Drive", and copy it from the mounted Drive to where the notebook looks for the archive (`shutil.copy("/content/drive/MyDrive/sp1500_earningscall.zip", "/content/MAEC_audio.zip")`, or into `ARCHIVE_DIR` as `MAEC_audio.zip`); the download cell then unpacks it without downloading. To depend on these downloads only once, set `ARCHIVE_DIR` in the settings cell of notebooks 1 and 2 to a folder on your Drive (for example `/content/drive/MyDrive/qa-grounded-volatility/archives`): the archives are then kept there (about 24 GB: 19.6 GB for MAEC and 4.7 GB for EC, so a free 15 GB Google Drive is too small) and every later session unpacks them without downloading. For notebook 2 you can also download the archives in a browser and unpack them yourself into `AUDIO_DIR/MAEC_audio` and `AUDIO_DIR/original_dataset`; the notebook uses audio that is already unpacked there.
