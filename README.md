# QNX ECG Beat Classification Research Demo

A tiny 1D-CNN ECG beat classifier running **natively on QNX SDP 8.0** (x86_64, QEMU
quick-start target). Replays a real MIT-BIH Arrhythmia Database record, classifies
each heartbeat into the five AAMI EC57 classes (N/S/V/F/Q), and prints the
prediction vs. the cardiologist annotation together with measured on-target
inference latency.

**Research/education demo only. Not a medical device. No diagnostic use.**

Built for the QNX Global Hackathon (category: Robotics & Medical). AI assistance
used during development (documented below).

## Why QNX

QNX is a real-time microkernel OS used in medical devices, automotive and
industrial systems. This project shows a complete on-target ML inference path
with no external dependencies: a portable C99 engine (`qnxapp/ecgnet.h`) that
runs bit-exact with the PyTorch training reference, built with `qcc` and run on
a live QNX 8.0 system.

## Repository layout

- `qnxapp/` - the QNX/host C application (inference engine, demo, parity test)
- `train/train_ecg.py` - full training + export recipe (PyTorch, CPU is fine)
- `weights/weights.bin` - trained float32 weights (131 KB)
- `data/README.md` - how to get the demo record `rec200.bin` (v1.0.0 release asset, or regenerate)
- `metrics.json` - verified evaluation results

## Model

Deliberately tiny: Conv1d(1->8,k5) MaxPool2, Conv1d(8->16,k5) MaxPool2,
Conv1d(16->32,k5) MaxPool2, Linear(608->48) ReLU, Linear(48->5). Input is a
180-sample window (0.5 s at 360 Hz) around each annotated R-peak, per-beat
z-normalized. No BatchNorm/Dropout so host<->target numerical parity is exact.

## Evaluation (honest, exploratory numbers)

Protocol: AAMI EC57 patient-disjoint DS1/DS2 split (de Chazal et al. 2004);
early stopping and the logit bias used held-out DS1 patients only
(records 215/220/223/230). IMPORTANT: DS2 was examined across more than one
training iteration during development (a tuning variant was rejected after
seeing DS2 regress), so DS2 is an EXPLORATORY test set here, not an untouched
final holdout. Treat these numbers as exploratory evidence, not a validated
generalization claim.

| Metric | Value | Baseline |
|---|---|---|
| Overall accuracy | 80.2% | 89.0% (majority class) |
| Balanced accuracy | 48.6% | 20% (chance) |
| Macro-F1 | 0.399 | 0.163 (majority) |

Per-class recall (DS2): N 0.82, S 0.75, V 0.73, F 0.14, Q 0.00.
Known limits: Q (paced) has 8 training / 7 test beats and is not learnable from
this data; F (fusion) is weak. Under class imbalance the majority baseline beats
raw accuracy - this model's value is minority-class recall, not the headline
number. See metrics.json and MODEL_CARD.md.

Host<->C parity: the C engine classifies 5000 exported DS2 beats with exactly
the same predictions as PyTorch (3982/5000 both) - this verifies the port, not
new accuracy. The demo record (data/rec200.bin, distributed as a release asset) is a DS2 member: its
91% replay score illustrates function on familiar data and is NOT independent
accuracy evidence. Host latency (~115 us/beat on the dev machine) is host
timing, not QNX target timing.

## On-target run (QNX 8.0)

Verified on a live QNX SDP 8.0.0 system (QSTI qemu quick-start image, x86_64,
TCG emulation): `uname -a` -> "QNX qnxqemu 8.0.0 2026/06/05-16:26:03EDT x86pc
x86_64". The qcc-built binary replayed record 200 (2600 beats) with per-beat
inference timing via CLOCK_MONOTONIC: 91.2% beat-level agreement (matches the
host run exactly, as the parity check predicts), avg 1.88 ms/beat in emulation
(~115 us on the dev host; the difference is emulation overhead, not QNX). This
is the demo app's own functional timing, not a QNX benchmark. Full captured
output: docs/ontarget_full.txt; demo video: https://youtu.be/eY4Dd34nXLs (QNX ECG Beat Classification Research Demo - QNX Global Hackathon).

## Build

Host smoke test (any POSIX cc):
    cc -O2 -o ecg_demo qnxapp/ecg_demo.c qnxapp/ecgnet.c -lm
    # first get data/rec200.bin: see data/README.md (release asset or train_ecg.py)
    ./ecg_demo --fast --record data/rec200.bin --weights weights/weights.bin

QNX 8.0 target (requires QNX SDP 8.0):
    . ~/qnx800/qnxsdp-env.sh
    qcc -Vgcc_ntox86_64 -O2 -o ecg_demo_qnx qnxapp/ecg_demo.c qnxapp/ecgnet.c -lm
    # copy ecg_demo_qnx + weights.bin + rec200.bin to the target, then run

## Licenses

- Code: MIT (see LICENSE).
- Model weights: MIT, trained on MIT-BIH.
- MIT-BIH Arrhythmia Database: PhysioNet, ODC-BY 1.0 (see NOTICE for attribution).
- QNX SDP 8.0 used under the QNX Non-Commercial License (not redistributed here).
