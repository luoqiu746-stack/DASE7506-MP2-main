# DASE7506 MP2 - E44 code submission

Final frozen Test: **393/467 = 84.1541755888651%**.
Matching implementation commit: `7d633199073d36020b0298440fa7d753c846f05e`.
The source files are byte-identical to this frozen E44 version. `SOURCE_SHA256.json` records their hashes.

## Contents

- `code/`: prediction implementation, training and analysis scripts, official evaluation tools and pinned dependencies.
- `report/MP2_Report.pdf`: 8-page English report, with AI assistance/reuse disclosure; editable Word version alongside it. Author: Luo Xinqiu. Student ID: 3036707989.
- `evidence/`: official Test JSON and predictions, measured process timings, experiment records, asset audit and freeze metadata.
- `evidence/E44_RESULTS.md`: historical detailed experimental results. Its original “uncommitted” wording describes when it was written; matching code was subsequently committed at the version above. Its historical Excel row count is superseded by the included current workbook (1952 rows including header).
- `guide/`: supplied assignment guide.

## Data and checkpoint

Obtain the official dataset from https://github.com/xudongwu-0/DASE7506-MP2 and place its `code/data/` contents under this package's `code/data/`. Data files are not duplicated here; supplied license, notice and manifest are preserved.

Download the separate `MP2_E44_Checkpoint.zip` and extract it into `code/runs/`. All nine inference assets must remain together. Expected location:

`code/runs/MP2_E44_Checkpoint/checkpoint/config.json`

The accompanying SHA256SUMS.json in the checkpoint package identifies the exact assets. The checkpoint is sufficient for inference without retraining.

## Install and reproduce frozen Test

Use Python 3.12. Run from this package's root:

```bash
cd code
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
.venv/bin/python check_data.py
.venv/bin/python predict.py --method submission.py \
  --checkpoint runs/MP2_E44_Checkpoint/checkpoint --split test \
  --output runs/MP2_E44_Checkpoint/test-predictions.json
.venv/bin/python score_predictions.py --split test \
  --predictions runs/MP2_E44_Checkpoint/test-predictions.json \
  --output runs/MP2_E44_Checkpoint/test-score.json
```

Expected score: `correct=393`, `n=467`, `accuracy=84.1541755888651`.
Local measured full-process wall time was 1.7645867919782177 s prediction + 0.057746791979297996 s independent scoring = 1.8223335839575157 s on Apple M5 MacBook Air, 24 GB RAM, CPU only, four threads. Reference Xeon was not tested. Program-internal JSON/Excel times omit some startup overhead.

## Rebuild the training chain from official Train

Training is optional for reproducing the submitted score. From `code/` with the environment above, use new output directories that have never been started or evaluated:

```bash
mkdir -p runs/e32-clean runs/e36-clean runs/e40-clean runs/e42-clean runs/e44-clean
.venv/bin/python -u train_scorer.py --expanded --run-dir runs/e32-clean
.venv/bin/python -u analyze_e22.py --select-expanded runs/e32-clean
.venv/bin/python -u train_scorer.py --kernel --run-dir runs/e36-clean --parent-run runs/e32-clean
.venv/bin/python -u analyze_e22.py --select-kernel runs/e36-clean
.venv/bin/python -u train_scorer.py --process --run-dir runs/e40-clean --parent-run runs/e36-clean
.venv/bin/python -u analyze_e22.py --select-process runs/e40-clean
.venv/bin/python -u train_scorer.py --stability --run-dir runs/e42-clean --parent-run runs/e40-clean
.venv/bin/python -u analyze_e22.py --select-stability runs/e42-clean
.venv/bin/python -u train_scorer.py --repair --run-dir runs/e44-clean --parent-run runs/e42-clean
.venv/bin/python -u analyze_e22.py --select-repair runs/e44-clean
```

Stages are expanded features, kernel model, process features, stability ensemble, and arithmetic repair. Each training stage fits only official Train; selection uses Train OOF and official Validation before freezing. Selection commands also produce a frozen Test report (or inherit the frozen parent result); never use those Test outcomes for tuning or selection. Existing internal-development and Validation data have been used repeatedly, as disclosed in the report. Retraining was performed during development; this code-packaging step did not rerun it. Runtime and model-file metadata may differ between machines.

## Publication

Upload the extracted source, report and evidence to a publicly accessible repository for the Code link. Publish `MP2_E44_Checkpoint.zip` as a downloadable Release asset for the Checkpoint link. This local package has not been pushed or externally submitted automatically. The source commit above identifies the implementation; a later documentation-only publishing commit may contain this README and report as well.
