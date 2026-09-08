# GNN-BERT Music Context Understanding

Complete, Colab-ready implementation for the CSE425/EEE474/CSE715 supervised neural-network project. It implements the required BERT baseline, audio graph GNN, CNN baseline, GNN-BERT fusion ablations, evaluation plots, graph examples, t-SNE, case studies, and an optional contrastive extension.

## Chosen experimental design

- **Dataset:** FMA-small audio plus the paired official FMA metadata.
- **Text input:** track title, album title, and artist name only.
- **Targets:** the eight balanced FMA-small top-level genres, represented as a multi-label vector with one positive label per track. The code also supports true multi-label `genres_all` mode.
- **Graph:** one node per 5-second segment. Node features combine log-mel, chroma, MFCC, spectral, RMS, and zero-crossing summaries. Edges join adjacent segments and acoustically similar segments.
- **Leakage control:** target genre names are never inserted into text. The official FMA train/validation/test split is retained. A split audit reports artist overlap.
- **Required comparisons:** random/majority, CNN, BERT-only, GNN-only, early concatenation, and cross-attention fusion.
- **Bonus:** symmetric InfoNCE dual-encoder training and caption/audio-style retrieval metrics.

FMA-small is practical on Colab and already pairs audio, metadata text, and genre annotations by track ID. The official dataset repository describes the metadata and provides the download files: <https://github.com/mdeff/fma>.

## Start here: Google Colab

1. Download this project ZIP and upload it to Google Drive.
2. Open `notebooks/CSE425_Full_Project_Colab.ipynb` in Colab.
3. Select **Runtime > Change runtime type > T4 GPU**.
4. Run cells in order. First run the smoke test, then switch `RUN_PROFILE` from `quick` to `final`.
5. Download the generated `submission_bundle.zip` after the final cell.

The notebook performs installation, download, preprocessing, training, evaluation, visualization, graph-sample export, and packaging. Training results are not invented in this repository: the report table is filled from the metrics produced by your actual run.

## Local/terminal workflow

```bash
python -m pip install -r requirements.txt
python -m src.download_data --config config.yaml --dataset fma_small
python -m src.preprocess --config config.yaml --profile quick
python -m src.audit_splits --config config.yaml
python -m src.run_all --config config.yaml --profile quick
python -m src.export_results --config config.yaml
```

Use `--profile final` for the complete 8,000-track run. `quick` limits the number of tracks and epochs so that errors can be found cheaply; it is not the final experiment.

## Main outputs

| Requirement | Generated output |
|---|---|
| 20+ graph samples | `data/processed/samples/*.json` |
| Training curves | `plots/*_training_curves.png` |
| F1 and AUC-PR comparison | `results/metrics.json`, `plots/model_comparison.png` |
| t-SNE | `plots/fusion_tsne.png` |
| Three graph/text case studies | `plots/case_study_*.png`, `results/case_studies.json` |
| Optional retrieval | `results/retrieval_metrics.json`, `retrieval_examples/` |
| Demo | `notebooks/demo_context.ipynb` |
| 6-10 page report source | `report/main.tex` |

## Important submission rule

Do not submit the placeholder report table before a real final run. Compile `report/main.tex` only after `results/metrics.json` and plots exist. The Colab notebook copies the numeric results into a LaTeX table automatically.

## Repository map

```text
src/audio_features.py   audio loading, log-mel/chroma/MFCC extraction
src/graph_builder.py    segment graph construction
src/fma.py              FMA metadata and leakage-safe examples
src/datasets.py         PyTorch/PyG datasets and collators
src/bert_encoder.py     Task 1
src/cnn_model.py        CNN baseline
src/gnn_model.py        Task 2
src/fusion_model.py     Task 3 early-concat and cross-attention
src/contrastive.py      Task 4 bonus
src/train.py            one-model training entry point
src/evaluate.py         metrics and plots
src/run_all.py          full ablation runner
src/export_results.py   final tables, plots, examples, and bundle
```

See `docs/REQUIREMENTS_MAPPING.md`, `docs/COLAB_GUIDE.md`, and `docs/SUBMISSION_CHECKLIST.md` before submission.

