# CI-DRL-Disease-prediction

## Human-in-the-Loop Deep Reinforcement Learning for Clinical Disease Prediction

### Overview

This module implements a Human-in-the-Loop (HITL) Deep Reinforcement Learning (DRL) system for disease classification from patient vitals and reported symptoms. It was developed as a component of a larger distributed clinical decision-support system, where model inference and doctor review occur as part of a coordinated pipeline rather than as an isolated, static classifier.

The central design principle is that disease prediction is not treated as a fixed, offline supervised learning problem. The model is formulated as a policy that selects a diagnosis given a patient's state, is trained with policy-gradient reinforcement learning, and continues to be updated in production using corrective feedback supplied by a reviewing physician. Every doctor review therefore functions as a live training signal, allowing the model's decision policy to be refined continuously against ground truth established by clinical judgment, rather than relying solely on a fixed training corpus.

### Motivation

Conventional supervised classifiers for clinical triage are trained once and deployed statically. Their accuracy is bounded by the training distribution, and any systematic errors persist until the model is retrained and redeployed through a separate offline pipeline. In a distributed clinical setting, this introduces two problems: predictions cannot be corrected in real time, and clinician expertise gathered at the point of care is not captured as a reusable training signal.

This system addresses both problems by:

1. Framing prediction as a reinforcement learning policy rather than a static classifier, allowing incremental weight updates from individual feedback events.
2. Embedding the physician directly in the inference loop, so that a reviewed and confirmed or corrected prediction immediately informs the deployed model.
3. Weighting the reward signal by clinical severity, so that errors on higher-acuity conditions are penalized more heavily than errors on lower-acuity or common conditions.

### System Architecture

The pipeline consists of four stages, corresponding to four scripts in this repository:

| Stage | Script | Function |
|---|---|---|
| Data preparation | `prepare_dataset.py` | Transforms raw patient records into a structured, model-ready feature set |
| Training | `main_train.py` | Supervised warm-start followed by policy-gradient fine-tuning |
| Evaluation | `test.py` | Offline evaluation of the trained policy against a held-out split |
| Inference with feedback | `predict.py` | Serves predictions to a clinician and applies a live policy update from their feedback |

#### 1. Data Preparation (`prepare_dataset.py`)

Raw patient records (`raw_health.csv`) are transformed into a cleaned, numeric feature table (`lab_health.csv`) suitable for model consumption:

- `Blood_Pressure_mmHg` is parsed and split into separate `Systolic_BP` and `Diastolic_BP` fields.
- `Gender` is encoded as a binary numeric field.
- Reported symptoms are consolidated into a fixed vocabulary of eight binary indicator columns: Cough, Fever, Fatigue, Shortness of breath, Runny nose, Headache, Body ache, and Sore throat.
- Records with missing essential vitals are removed to preserve data integrity for downstream training.
- The diagnosis field is renamed to `Disease` and retained as the prediction target.

The resulting dataset comprises 2,000 labeled patient records across five diagnostic classes: Healthy, Bronchitis, Flu, Cold, and Pneumonia. Class distribution is imbalanced, with Healthy and Bronchitis substantially overrepresented relative to Pneumonia, which is addressed during training.

#### 2. Model and Training Pipeline (`main_train.py`)

**State and action space.** Each patient record is represented as a 15-dimensional feature vector (seven vital-sign and demographic features plus eight binary symptom indicators). The action space corresponds to the five diagnostic classes.

**Policy network (`ActorNet`).** A feed-forward neural network with two hidden layers of 128 units each (ReLU activations), mapping the patient state vector to a categorical distribution over diagnoses.

**Stage 1 — Supervised warm-start.** Prior to reinforcement learning, the policy network is pretrained using standard supervised learning to establish a stable initial parameterization:

- Class imbalance in the training split is corrected using SMOTE (Synthetic Minority Over-sampling Technique) prior to fitting.
- The network is optimized with Focal Loss (alpha = 1.0, gamma = 2.0) rather than standard cross-entropy, which down-weights well-classified majority-class examples and directs learning capacity toward harder, typically minority-class, cases.
- Features are standardized using `StandardScaler`, fitted on the training split and persisted for consistent use at inference time.
- Training uses early stopping on validation accuracy (patience of 8 epochs, maximum 60 epochs) to prevent overfitting.

**Stage 2 — Policy-gradient fine-tuning (REINFORCE).** Following the warm-start, the network is treated as a stochastic policy and fine-tuned using the REINFORCE algorithm:

- A lightweight environment (`DatasetEnv`) presents patient states sequentially and returns a reward upon each diagnostic action.
- Rewards are class-weighted rather than uniform: correct predictions receive a baseline reward, with elevated rewards assigned to more clinically significant conditions (for example, a higher reward for correctly identifying Pneumonia than for Healthy), while incorrect predictions receive a fixed negative reward. This design directs the policy toward minimizing clinically costly errors rather than optimizing for raw accuracy alone.
- Returns are computed with a discount factor of 0.99 over episodes of 200 steps, and gradient updates use a moving-average baseline (momentum 0.99) to reduce variance in the policy gradient estimate.
- The best-performing checkpoint by validation accuracy is retained separately (`pg_drl_model_best.pth`) from the final-epoch checkpoint (`pg_drl_model_final.pth`).

#### 3. Evaluation (`test.py`)

The retained best checkpoint is evaluated against a held-out test split (`X_test_pg.npy`, `y_test_pg.npy`) that is excluded from both the supervised and reinforcement learning stages. Evaluation reports overall accuracy, a per-class precision/recall/F1 classification report, and a confusion matrix, providing visibility into class-specific performance given the underlying class imbalance.

#### 4. Human-in-the-Loop Inference (`predict.py`)

This component implements the clinical feedback loop and is the primary interface between the model and the reviewing physician:

1. Patient vitals (age, gender, heart rate, body temperature, oxygen saturation, systolic and diastolic blood pressure) and up to three reported symptoms are collected as input.
2. The current policy produces a ranked list of the three most probable diagnoses with associated confidence scores.
3. The physician reviews the top prediction and provides a binary confirmation.
   - If confirmed, the action is reinforced with a positive reward.
   - If rejected, the physician supplies the correct diagnosis, and the action is assigned a negative reward.
4. A REINFORCE gradient step is applied immediately to the deployed model using the observed action and the physician-derived reward signal, and the updated weights are persisted to `pg_drl_model_best.pth`.

This mechanism ensures that physician corrections are incorporated into the model's decision policy at the point of care, without requiring a separate retraining cycle.

### Repository Contents

```
raw_health.csv            Raw patient records
lab_health.csv            Cleaned, model-ready dataset
prepare_dataset.py        Data cleaning and feature engineering
main_train.py             Supervised warm-start and REINFORCE training pipeline
test.py                   Offline evaluation on the held-out test split
predict.py                HITL inference and live feedback-driven policy update
pg_drl_model_best.pth     Best checkpoint by validation accuracy
pg_drl_model_final.pth    Final checkpoint after training completion
scaler_pg.pkl             Fitted StandardScaler for input normalization
labelenc_pg.pkl           Fitted LabelEncoder for diagnostic classes
symptom_list.csv          Reference symptom vocabulary with observed frequencies
X_test_pg.npy             Held-out test features
y_test_pg.npy             Held-out test labels
```

### Role Within the Distributed System

Within the broader distributed system, this module functions as a decision-support microservice: it accepts patient state as input, returns a ranked prediction with confidence scores, and exposes an update pathway through which physician feedback is applied to the deployed policy. This separation allows the reinforcement learning component to operate independently of upstream data ingestion and downstream reporting services, while remaining responsive to corrective signals generated elsewhere in the system.

### Limitations and Considerations

- The dataset used for training and evaluation is limited in scale and class balance, with Pneumonia in particular underrepresented; reported performance should be interpreted in that context.
- Because each feedback event triggers an immediate, single-example gradient update, the deployed policy can shift after a small number of corrections. In a production distributed deployment, this should be accompanied by update logging, rate-limiting or batching of feedback-driven updates, and periodic re-validation against the held-out set to detect and control for drift introduced by noisy or inconsistent feedback.
- The system is designed as a decision-support aid intended to operate under continuous physician oversight. It is not intended to produce unreviewed diagnoses and has not been validated for autonomous clinical use.
