# Model card - tiny ECG beat CNN (research demo)

- Task: AAMI EC57 5-class ECG beat classification (N/S/V/F/Q), 180-sample
  window at 360 Hz, per-beat z-normalized. Lead II (MLII) of MIT-BIH.
- Data: MIT-BIH Arrhythmia Database (PhysioNet, ODC-BY 1.0, see NOTICE).
  DS1 train + DS1 val (records 215/220/223/230, patient-disjoint early stop).
- Recipe: natural sampling, weighted cross-entropy (weights clipped at 15),
  AdamW lr 3e-4 wd 1e-3, batch 256, early stop on val balanced accuracy,
  logit bias {0.5,0,0,0,0}. Frozen; no further test-driven tuning.
- Status of metrics: EXPLORATORY. DS2 was inspected across training
  iterations, so it is not an untouched holdout. No independent
  generalization claim is made.
- DS2 (exploratory): overall 80.2% (majority baseline 89.0%), balanced 48.6%,
  macro-F1 0.399. Per-class recall N .82 / S .75 / V .73 / F .14 / Q .00.
- Known limitations: Q class (8 train / 7 test beats) not learnable; F weak;
  overall accuracy below majority baseline under imbalance; single-lead,
  single-database; no noise/artifact robustness testing; not validated on
  any other dataset or population.
- Intended use: hackathon research/education demo of on-target QNX inference.
  NOT a medical device; no diagnostic, screening or monitoring use.
- AI assistance: model/code developed with AI-assistant help; all metrics
  measured by the scripts in this repo.
