# Demo record (rec200.bin)

`ecg_demo` replays a demo record, `rec200.bin` (MIT-BIH Arrhythmia Database
record 200, 2600 annotated beats, ODC-BY 1.0 - see ../NOTICE).

Get it either way:

1. Download the prebuilt binary from the v1.0.0 release assets:
   https://github.com/mjanthony819/qnx-ecg-beat-classification/releases/download/v1.0.0/rec200.bin
   SHA-256: 863b6c23d352d44ec49f4167a24e537cee5588cce55405478a61b56f0914a0ca

2. Regenerate it from source: run `train/train_ecg.py` (it reads MIT-BIH from
   PhysioNet and exports `rec200.bin` alongside `weights.bin`).

Note: record 200 is a member of the exploratory DS2 evaluation set (see
../README.md), so its replay score illustrates function on familiar data, not
independent accuracy.
