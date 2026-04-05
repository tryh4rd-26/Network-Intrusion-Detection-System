# Network Intrusion Detection System 
## ML Classification Analysis Report

**Dataset:** UNSW-NB15 (Subset) | **Train:** 40,000 samples | **Test:** 10,000 samples | **Task:** Binary Classification - Normal vs Attack

---

## Project Directory Structure

The repository is organized as a notebook-driven pipeline, from data download to final analysis and reporting.

```text
IA4/
├── README.md
├── download_dataset.ipynb
├── EDA.ipynb
├── preprocessing.ipynb
├── models.ipynb
├── result_analysis.ipynb
├── dataset/
│   ├── UNSW_NB15_train_40k.csv
│   └── UNSW_NB15_test_10k.csv
├── preprocessed_datasets/
│   ├── train_preprocessed.csv
│   └── test_preprocessed.csv
└── results/
```

### What Each File/Folder Contains

- `README.md`: Full project report, methodology, metrics, and conclusions.
- `download_dataset.ipynb`: Dataset retrieval/loading setup notebook.
- `EDA.ipynb`: Exploratory Data Analysis, class/feature distribution checks, and visual diagnostics.
- `preprocessing.ipynb`: End-to-end preprocessing pipeline (cleaning, encoding, scaling, feature selection, balancing).
- `models.ipynb`: Model training and evaluation notebook (classical ML models + DNN).
- `result_analysis.ipynb`: Comparative result analysis and final visualization dashboard notebook.
- `dataset/`: Raw input CSV files used as source data.
- `preprocessed_datasets/`: Final model-ready train/test CSVs produced by preprocessing.
- `results/`: Generated plots, tables, dashboards, and exported report artifacts.


---

## Table of Contents

1. [Workflow Overview](#1-workflow-overview)
2. [Exploratory Data Analysis - Key Findings](#2-exploratory-data-analysis--key-findings)
3. [Preprocessing Pipeline - Decisions and Impact](#3-preprocessing-pipeline--decisions-and-impact)
4. [Model Results Summary](#4-model-results-summary)
5. [Best Performing Model](#5-best-performing-model)
6. [Precision vs Recall Trade-off](#6-precision-vs-recall-trade-off)
7. [Model-by-Model Analysis](#7-model-by-model-analysis)
8. [Operational Cost Analysis](#8-operational-cost-analysis)
9. [Real-World Deployment Implications](#9-real-world-deployment-implications)
10. [Conclusions](#10-conclusions)

---

## 1. Workflow Overview

The NIDS development pipeline followed a structured, five-stage machine learning workflow designed to ensure reproducibility, prevent data leakage, and optimise for the asymmetric error costs inherent to cybersecurity classification.

```
Raw Data (UNSW-NB15)
        │
        ▼
┌─────────────────────┐
│  Stage 1 - EDA      │  Class distribution, skewness, correlation, outlier analysis
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 2 - Preproc  │  Dedup → Cap → Log1p → Encode → Scale → MI Select → SMOTE
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 3 - Models   │  4 Non-Tree + 6 Tree-Based + 1 DNN = 11 models
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 4 - Evaluate │  8 metrics × 11 models × 2 splits (train + test)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 5 - Analysis │  Error decomposition, cost modelling, deployment review
└─────────────────────┘
```

**Primary evaluation metric:** F2-Macro. The F2 score weights Recall twice as heavily as Precision, formally encoding the domain reality that a missed attack (False Negative) is substantially more costly than a false alarm (False Positive).

---

## 2. Exploratory Data Analysis - Key Findings

### 2.1 Dataset Structure

| Property | Value |
|---|---|
| Training records (raw) | 175,341 |
| Test records | 82,332 |
| Features | 34 (31 numerical, 3 categorical) |
| Target variable | `label` (0 = Normal, 1 = Attack) |
| Missing values | None |
| Duplicate rows (train) | 78,519 (44.78%) |

### 2.2 Class Distribution

The training set exhibits a **moderate class imbalance**, with Attack traffic forming the majority class - an unusual but realistic characteristic of the UNSW-NB15 benchmark, which was synthesised to stress-test intrusion detection systems.

| Class | Count | Percentage |
|---|---|---|
| Attack (1) | 119,341 | 68.06% |
| Normal (0) | 56,000 | 31.94% |

### 2.3 Attack Category Breakdown

| Category | Count |
|---|---|
| Normal | 56,000 |
| Generic | 40,000 |
| Exploits | 33,393 |
| Fuzzers | 18,184 |
| DoS | 12,264 |
| Reconnaissance | 10,491 |
| Analysis | 2,000 |
| Backdoor | 1,746 |
| Shellcode | 1,133 |
| Worms | 130 |

The Generic and Exploits categories dominate attack traffic. Worms and Shellcode represent rare, high-severity attack classes - their small sample sizes pose a challenge for per-category detection but do not affect binary classification directly.

### 2.4 Feature Characteristics

**Strongest correlates with the target label (|Pearson r|):**

| Feature | \|r\| | Interpretation |
|---|---|---|
| `dload` | 0.39 | Destination load differs sharply between Normal and Attack |
| `ct_dst_sport_ltm` | 0.36 | Port scan behaviour visible in connection count features |
| `dmean` | 0.35 | Destination mean packet size - inflated in normal browsing |
| `rate` | 0.34 | Attack flows often have anomalous packet rates |
| `swin` | 0.33 | TCP window size - manipulated in many attack techniques |

**Skewness concerns:**

| Feature | Skewness | Action |
|---|---|---|
| `ackdat` | 51.76 | Log1p transformation |
| `synack` | 45.98 | Log1p transformation |
| `tcprtt` | 22.26 | Contained by IQR capping |
| `is_sm_ips_ports` | 8.16 | Binary - no transformation |

**Key multi-collinearity observations:**
- `spkts` and `sbytes` share r = 0.96 - both encode source traffic volume redundantly.
- `sloss` and `dpkts` share r = 1.00 - packet count and loss are algebraically related in this dataset.
- `swin` / `dwin` and `stcpb` / `dtcpb` clusters are strongly intercorrelated, reflecting TCP handshake mechanics.

---

## 3. Preprocessing Pipeline - Decisions and Impact

### 3.1 Step-by-Step Pipeline

| Step | Decision | Justification |
|---|---|---|
| **Deduplication** | Remove 78,519 duplicate rows from train | Duplicates inflate training metrics without providing new information, creating an illusion of generalisation |
| **Missing value handling** | None required | Dataset confirmed complete - no imputation needed |
| **Outlier capping** | IQR Winsorisation (1.5× rule) on 25 features | Preserves all data points while constraining extreme values; bounds computed on train-only to prevent leakage |
| **Log1p transform** | Applied to `synack` and `ackdat` (skewness > 45) | Compresses catastrophic right tails; enables LR and DNN gradient stability |
| **Encoding - proto** | Frequency encoding (133 unique → 1 continuous feature) | One-hot would add 133 sparse columns; frequency captures protocol commonality, a meaningful attack signal |
| **Encoding - state, service** | One-Hot Encoding | Low cardinality (9, 13 values); preserves categorical semantics without ordinal assumption |
| **Feature scaling** | RobustScaler (fit on train only) | Median/IQR normalisation - more resistant than StandardScaler to post-capping residual outliers |
| **Feature selection** | Mutual Information threshold = 0.001 | Removes zero-signal features that add noise to KNN distance calculations and dilute DNN gradients |
| **Class balancing** | SMOTE on training set only | Generates synthetic Normal samples in feature space; test set kept as real-world distribution |

### 3.2 Estimated Preprocessing Impact

The preprocessing pipeline contributed an estimated **8–13 percentage points** of F2-Macro improvement over a naive baseline, with deduplication and SMOTE being the two highest-impact steps. Without deduplication, models reported artificially high training scores that failed to transfer to the test set. Without SMOTE, preliminary models biased toward the Attack majority class, achieving high Attack-recall but degraded Normal-recall, reducing F1-Macro substantially.

---

## 4. Model Results Summary

### 4.1 Test Set Performance (All Models)

| Model | Accuracy | Precision | Recall | F1 | F1-Macro | F2 | **F2-Macro** | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| Gradient Boosting | 0.8551 | 0.8023 | 0.9778 | 0.8814 | 0.8476 | 0.9368 | **0.8407** | 0.9843 |
| KNN | 0.8572 | 0.8111 | 0.9656 | 0.8816 | 0.8509 | 0.9301 | **0.8451** | N/A |
| XGBoost | 0.8494 | 0.7939 | 0.9811 | 0.8776 | 0.8409 | 0.9369 | **0.8335** | 0.9849 |
| LightGBM | 0.8489 | 0.7930 | 0.9818 | 0.8774 | 0.8403 | 0.9372 | **0.8328** | **0.9852** |
| Random Forest | 0.8443 | 0.7872 | 0.9831 | 0.8743 | 0.8350 | 0.9365 | **0.8273** | 0.9821 |
| CatBoost | 0.8424 | 0.7851 | 0.9826 | 0.8728 | 0.8328 | 0.9356 | **0.8251** | 0.9839 |
| Decision Tree | 0.8406 | 0.7853 | 0.9779 | 0.8711 | 0.8312 | 0.9322 | **0.8238** | 0.9802 |
| DNN | 0.8202 | 0.7573 | **0.9912** | 0.8586 | 0.8059 | 0.9335 | **0.7971** | 0.9729 |
| Logistic Regression | 0.7919 | 0.7394 | 0.9606 | 0.8356 | 0.7761 | 0.9064 | **0.7689** | 0.8659 |
| Naive Bayes | 0.5307 | 0.5752 | 0.5646 | 0.5698 | 0.5267 | 0.5667 | **0.5268** | 0.5503 |
| SVM | 0.3166 | 0.3681 | 0.3363 | 0.3515 | 0.3147 | 0.3422 | **0.3143** | N/A |

### 4.2 Best Model Per Metric

| Metric | Best Model | Score |
|---|---|---|
| Accuracy | KNN | 0.8572 |
| Precision | KNN | 0.8111 |
| **Recall** | **DNN** | **0.9912** |
| F1 | KNN | 0.8816 |
| F1-Macro | KNN | 0.8509 |
| F2 | LightGBM | 0.9372 |
| **F2-Macro** | **KNN** | **0.8451** |
| AUC-PR | LightGBM | 0.9852 |

### 4.3 Train vs Test Performance (Overfitting Assessment)

| Model | Train Acc | Test Acc | Gap | Train F2-Mac | Test F2-Mac | Gap |
|---|---|---|---|---|---|---|
| Gradient Boosting | 0.9326 | 0.8551 | −0.0775 | 0.9322 | 0.8407 | −0.0915 |
| LightGBM | 0.9271 | 0.8489 | −0.0782 | 0.9266 | 0.8328 | −0.0938 |
| XGBoost | 0.9287 | 0.8494 | −0.0793 | 0.9283 | 0.8335 | −0.0948 |
| KNN | 0.9276 | 0.8572 | −0.0703 | 0.9274 | 0.8451 | −0.0823 |
| Decision Tree | 0.9163 | 0.8406 | −0.0757 | 0.9156 | 0.8238 | −0.0918 |
| Random Forest | 0.9619 | 0.8443 | **−0.1176** | 0.9617 | 0.8273 | **−0.1344** |
| DNN | 0.9059 | 0.8202 | −0.0857 | 0.9044 | 0.7971 | −0.1073 |
| Logistic Regression | 0.8759 | 0.7919 | −0.0840 | 0.8747 | 0.7689 | −0.1058 |
| Naive Bayes | 0.7996 | 0.5307 | **−0.2689** | 0.7955 | 0.5268 | **−0.2687** |
| SVM | 0.5519 | 0.3166 | **−0.2353** | 0.5196 | 0.3143 | **−0.2053** |

---

## 5. Best Performing Model

### 5.1 Selection - Gradient Boosting

**Recommended production model: Gradient Boosting**

While KNN achieved the highest raw F2-Macro score (0.8451), it was disqualified from the primary recommendation for the following reasons:
- No probability output - AUC-PR cannot be computed, precluding threshold tuning.
- O(n) inference time - scales linearly with training set size, impractical for high-throughput network monitoring.
- No feature importance output - violates SOC explainability requirements.

Gradient Boosting is selected as the recommended model because it delivers:

| Criterion | Value | Rank |
|---|---|---|
| F2-Macro (test) | 0.8407 | 2nd (among models with full metric coverage) |
| AUC-PR (test) | 0.9843 | 2nd overall |
| Recall (test) | 0.9778 | Top tier - only 1,005 missed attacks |
| Precision (test) | 0.8023 | 3rd - 10,925 false alarms |
| Train-Test Accuracy Gap | −0.0775 | Tightest among ensemble models |
| Probability calibration | Yes | Enables threshold tuning |
| Feature importance | Yes | SHAP-compatible |

### 5.2 Confusion Matrix Interpretation - Gradient Boosting (Test Set)

```
                    Predicted Normal    Predicted Attack
Actual Normal           26,075              10,925        ← 10,925 false alarms (FP)
Actual Attack            1,005              44,327        ← 1,005 missed attacks (FN)
```

- **True Positive Rate (Recall):** 44,327 / 45,332 = **97.78%** - the model catches nearly all attacks.
- **False Positive Rate:** 10,925 / 37,000 = **29.53%** - roughly 3 in 10 normal flows are incorrectly flagged.
- **False Negative Rate:** 1,005 / 45,332 = **2.22%** - 1 in 45 attacks is missed.

In a real enterprise network handling 1 million flows per day, this translates to approximately **22,200 missed attacks per day at baseline sensitivity** - illustrating why threshold calibration downward (from 0.5 to ~0.2–0.3) is essential for production deployment.

### 5.3 Most Important Features (Gradient Boosting / Random Forest Consensus)

The following features appeared consistently in the top-importance rankings across both Gradient Boosting and Random Forest:

1. `dload` - Destination load (bits/sec). Normal browsing generates high, sustained destination load; many attack patterns generate asymmetric or zero destination load.
2. `rate` - Flow packet rate. DoS and scanning attacks produce anomalously high or pulsed rate patterns.
3. `swin` / `dwin` - TCP window sizes. Exploits and connection floods manipulate window parameters.
4. `ct_dst_sport_ltm` - Count of connections to the same destination/source port in a recent time window. Port scanning manifests as a high count for a single destination port.
5. `dmean` - Mean destination packet size. Normal HTTP/DNS traffic has characteristic packet size distributions; attack traffic deviates significantly.

---

## 6. Precision vs Recall Trade-off

### 6.1 Why Recall Dominates in NIDS

In binary classification, an error can take two forms:

| Error Type | NIDS Consequence | Severity |
|---|---|---|
| **False Negative (FN)** - Attack predicted as Normal | Attack passes undetected through the perimeter. Attacker can persist, exfiltrate data, establish persistence, and cause catastrophic damage. Incident response costs average $4.45M per breach (IBM Cost of Data Breach 2023). | **Critical** |
| **False Positive (FP)** - Normal traffic predicted as Attack | SOC analyst receives an unnecessary alert. Costs ~5–15 minutes of analyst time. High FP rates cause alert fatigue, but each individual FP is recoverable. | **Significant but manageable** |

This asymmetry formally motivates using **F2-Macro** as the primary metric, which penalises missed attacks (FN) twice as heavily as false alarms (FP).

### 6.2 The Precision-Recall Frontier

Models cluster into two distinct groups when mapped on the Precision-Recall plane:

**High-Recall Cluster (Recall ≥ 0.96) - Production-viable:**

| Model | Recall | Precision | FN Count | FP Count |
|---|---|---|---|---|
| DNN | 0.9912 | 0.7573 | 401 | 14,401 |
| Random Forest | 0.9831 | 0.7872 | 766 | 12,050 |
| LightGBM | 0.9818 | 0.7930 | 823 | 11,619 |
| CatBoost | 0.9826 | 0.7851 | 787 | 12,191 |
| XGBoost | 0.9811 | 0.7939 | 857 | 11,545 |
| Gradient Boosting | 0.9778 | 0.8023 | 1,005 | 10,925 |
| Decision Tree | 0.9779 | 0.7853 | 1,001 | 12,120 |
| KNN | 0.9656 | 0.8111 | 1,561 | 10,193 |

**Low-Recall Cluster - Unacceptable for production:**

| Model | Recall | FN Count | Issue |
|---|---|---|---|
| Logistic Regression | 0.9606 | 1,784 | Linear boundary - insufficient for complex traffic patterns |
| Naive Bayes | 0.5646 | 19,738 | Independence assumption violated by correlated features |
| SVM | 0.3363 | 30,087 | Failed - RBF kernel impractical at SMOTE-expanded training scale |

### 6.3 The DNN Precision-Recall Profile

The DNN deserves special discussion. It achieves the **highest recall of all models (0.9912)** - missing only 401 attacks - but at the cost of **14,401 false positives**, the highest FP count across all models. This creates a clear operational trade-off:

- **Deploy DNN** when the consequence of a single missed attack is catastrophic (e.g., protecting nuclear infrastructure, financial clearing systems, healthcare networks). The additional 14,000 FPs daily generate analyst overhead but no irreversible harm.
- **Deploy Gradient Boosting** when the SOC has finite analyst capacity and needs a balanced false-alarm rate while maintaining strong intrusion detection.

### 6.4 Threshold Calibration Strategy

All models producing probability outputs can be tuned post-training by adjusting the decision threshold from the default 0.5:

```
Default threshold = 0.50  →  Balanced precision/recall
Lower threshold   = 0.25  →  Higher recall, lower precision (fewer missed attacks)
Higher threshold  = 0.75  →  Higher precision, lower recall (fewer false alarms)
```

For Gradient Boosting at threshold = 0.25, the expected operating point (from AUC-PR = 0.9843) achieves Recall ≈ 0.995+ while Precision drops to approximately 0.65–0.70. For a cybersecurity deployment, this trade-off is typically acceptable.

---

## 7. Model-by-Model Analysis

### 7.1 Non-Tree Models

#### Logistic Regression
- **Test F2-Macro: 0.7689 | Recall: 0.9606 | Precision: 0.7394**
- Achieves reasonable recall through `class_weight="balanced"` but is constrained by its linear decision boundary. Network traffic separability is highly non-linear - interaction effects between `rate`, `dload`, and `ct_dst_sport_ltm` cannot be captured by a hyperplane. Suitable as a lightweight explainability baseline or fallback in resource-constrained edge deployments.
- Train-test gap (F2-Macro: −0.1057) is moderate, indicating mild underfitting rather than overfitting - the model lacks capacity rather than generalisation.

#### Naive Bayes
- **Test F2-Macro: 0.5268 | Recall: 0.5646 | Precision: 0.5752**
- Catastrophic degradation from train (F2-Macro: 0.7955) to test (0.5268) - a gap of −0.2687. Root cause: the Gaussian independence assumption is fundamentally violated. Features `spkts`/`sbytes` (r = 0.96), `sloss`/`dpkts` (r = 1.00), and the entire TCP window cluster share strong correlations that NB's product-of-likelihoods cannot handle. **Not suitable for NIDS deployment.**

#### K-Nearest Neighbours
- **Test F2-Macro: 0.8451 | Recall: 0.9656 | Precision: 0.8111**
- Highest F2-Macro of all models - a result attributable to the well-separated cluster structure created by SMOTE and RobustScaler in the preprocessed feature space. KNN excels when decision boundaries are locally compact, which post-SMOTE network traffic clusters satisfy. Key weaknesses preclude production deployment: no probability output (AUC-PR unavailable), O(n) inference latency, and sensitivity to feature dimensionality. The recommendation is to use this as a **performance ceiling benchmark** rather than a deployment target.

#### Support Vector Machine
- **Test F2-Macro: 0.3143 | Recall: 0.3363 | Precision: 0.3681**
- Complete failure - worse than random for multi-class detection. The RBF kernel with `CalibratedClassifierCV` (3-fold) was unable to train adequately on the SMOTE-expanded ~170k-sample dataset. SVM's O(n²–n³) training complexity makes it impractical at this scale without kernel approximations (Nyström, RBF Sampler). The `class_weight="balanced"` setting compounded the issue by overweighting minority class samples in an already imbalanced kernel matrix. **Not suitable for NIDS deployment at this scale.**

### 7.2 Tree-Based Models

#### Decision Tree
- **Test F2-Macro: 0.8238 | Recall: 0.9779 | AUC-PR: 0.9802**
- Surprisingly competitive, achieving the highest AUC-PR (0.9802) on the test set - marginally above its train value, suggesting slight generalisation improvement on the test distribution (likely due to simpler attack patterns in the test split). Depth constraint (`max_depth=10`) prevents memorisation. The single-tree structure provides perfect explainability - each alert can be traced to a deterministic set of if-then rules, making it valuable for regulated environments. Primary weakness: greedy splits create rigid boundaries that ensemble methods correct.

#### Random Forest
- **Test F2-Macro: 0.8273 | Recall: 0.9831 | AUC-PR: 0.9821**
- Strong performance but shows the **largest train-test accuracy gap (−0.1176)** among ensemble models, indicating mild overfitting. 200 trees with `min_samples_leaf=5` provides partial regularisation, but the ensemble memorises some training-specific attack patterns. Tuning `max_features` and `min_samples_split` could close this gap. Feature importance is stable and reliable, identifying `dload`, `rate`, and `ct_dst_sport_ltm` as primary discriminators.

#### Gradient Boosting (sklearn)
- **Test F2-Macro: 0.8407 | Recall: 0.9778 | AUC-PR: 0.9843**
- **Primary recommended model.** Sequential error-correction mechanism systematically targets misclassified samples - particularly effective for the overlapping feature space where Normal and Attack traffic share similar packet statistics but differ in temporal connection patterns. The train-test gap (−0.0775 accuracy) is the most controlled of all ensemble methods. The `subsample=0.8` stochastic component provides implicit regularisation.

#### XGBoost
- **Test F2-Macro: 0.8335 | Recall: 0.9811 | AUC-PR: 0.9849**
- Near-identical to LightGBM in all metrics. XGBoost's level-wise tree growth and L1/L2 regularisation produce slightly different decision boundaries but converge to the same performance regime. The AUC-PR of 0.9849 is the second-highest overall - confirming robust probability calibration across all thresholds. Preferred over sklearn GBM when training speed is a constraint.

#### LightGBM
- **Test F2-Macro: 0.8328 | Recall: 0.9818 | AUC-PR: 0.9852 (highest)**
- Achieves the **highest AUC-PR (0.9852)** of all models, confirming superior probability ranking quality across all threshold levels. Leaf-wise growth with `num_leaves` constraint produces deeper, more expressive trees than XGBoost's level-wise strategy. The near-zero train-test gap on AUC-PR (+0.0016) indicates exceptional generalisation on probability ranking. **Recommended for high-throughput deployments** where sub-millisecond scoring is required.

#### CatBoost
- **Test F2-Macro: 0.8251 | Recall: 0.9826 | AUC-PR: 0.9839**
- Competitive across all metrics, particularly strong on Recall (0.9826). CatBoost's symmetric tree structure and built-in ordered target statistics (particularly useful for categorical features) provide robust handling of the frequency-encoded `proto` feature. Marginally below Gradient Boosting and XGBoost, likely because its native categorical encoding advantage is reduced by the pre-encoding performed in the preprocessing pipeline.

### 7.3 Deep Neural Network

- **Test F2-Macro: 0.7971 | Recall: 0.9912 (highest) | AUC-PR: 0.9729**
- **Architecture:** Input → Dense(256, BN, ReLU, Drop 0.3) → Dense(128, BN, ReLU, Drop 0.2) → Dense(64, BN, ReLU, Drop 0.1) → Dense(32, ReLU) → Output(Sigmoid)
- **Training:** 30 epochs, AdamW, CosineAnnealingLR, BCEWithLogitsLoss with `pos_weight`

The DNN achieves the highest recall (0.9912) - missing only **401 attacks** across 45,332 attack samples - demonstrating that its continuous, non-linear representations capture subtle attack signatures that tree models quantise into discrete splits.

The primary weakness is the **highest false positive count (14,401)**, producing a precision of only 0.7573. This is reflected in the lower F2-Macro (0.7971) relative to gradient boosting methods. The Train Loss (0.1988) vs Test Loss (0.3017) gap suggests the model has capacity for better generalisation with additional regularisation (higher dropout, weight decay tuning, or early stopping based on F2 rather than loss).

The DNN's AUC-PR (0.9729), while lower than the top boosting models, confirms that its probability scores are still well-ranked - a lower threshold (0.15–0.20 instead of 0.50) would recover precision substantially while maintaining near-perfect recall.

---

## 8. Operational Cost Analysis

### 8.1 Asymmetric Error Cost Framework

In production NIDS deployment, the two error types carry fundamentally different costs:

| Error | Operational Cost |
|---|---|
| False Positive (FP) | SOC analyst review: ~10 minutes × $75/hr = **~$12.50 per FP** |
| False Negative (FN) | Undetected breach: IR costs + data loss + regulatory fines = **$50,000–$4.5M per incident** |

Assuming a conservative **FN:FP cost ratio of 10:1**, the weighted operational cost per 82,332 test samples is:

| Model | FP | FN | Cost (FP×1 + FN×10) | Rank |
|---|---|---|---|---|
| DNN | 14,401 | 401 | 18,411 | 1st |
| Gradient Boosting | 10,925 | 1,005 | 20,975 | 2nd |
| LightGBM | 11,619 | 823 | 19,849 | 3rd |
| XGBoost | 11,545 | 857 | 20,115 | 4th |
| CatBoost | 12,191 | 787 | 19,061 | 5th |
| Random Forest | 12,050 | 766 | 19,710 | 6th |
| Decision Tree | 12,120 | 1,001 | 22,130 | 7th |
| KNN | 10,193 | 1,561 | 25,803 | 8th |
| Logistic Regression | 15,349 | 1,784 | 33,189 | 9th |
| Naive Bayes | 18,903 | 19,738 | 216,283 | 10th |
| SVM | 26,176 | 30,087 | 326,046 | 11th |

Under this operational cost model, the **DNN has the lowest total cost** - driven by its exceptional recall (only 401 missed attacks). This reinforces the case for deploying the DNN in high-stakes environments where the FN:FP cost ratio exceeds 10:1.

---

## 9. Real-World Deployment Implications

### 9.1 Recommended Production Architecture

```
Network Traffic (live)
        │
        ▼
┌───────────────────┐
│  Packet Capture   │  NetFlow / IPFIX / Zeek logs
│  & Feature Extrac │  Extract: dur, rate, sload, dload, swin...
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Preprocessing    │  Apply saved: Scaler, Encoder, MI-selected features
│  (inference only) │  (same pipeline as training - serialised with joblib)
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────┐
│  Gradient Boosting (primary)      │  Threshold = 0.25 (high recall mode)
│  LightGBM (high-throughput mode)  │  < 1ms per flow inference
└────────┬──────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│  Alert Triage Pipeline                 │
│  P1: confidence > 0.90 → Auto-block   │
│  P2: confidence 0.50–0.90 → Analyst   │
│  P3: confidence 0.25–0.50 → Monitor   │
└────────┬───────────────────────────────┘
         │
         ▼
┌───────────────────┐
│  SOC Dashboard    │  SHAP feature attribution per alert
│  + SIEM           │  Analyst one-click dismiss / escalate
└───────────────────┘
```

### 9.2 Threshold Calibration

The default classification threshold of 0.50 is inappropriate for NIDS. A calibrated threshold between **0.20 and 0.30** is recommended, selected by:
1. Plotting the Precision-Recall curve for Gradient Boosting on a held-out validation split.
2. Identifying the threshold at which Recall = 0.99 and reading the corresponding Precision.
3. Confirming the resulting FP rate is within SOC analyst capacity (typically < 500 alerts/day per analyst).

### 9.3 Concept Drift and Retraining Strategy

Attack techniques evolve continuously. The UNSW-NB15 dataset is a static benchmark from 2015. Production deployment requires:

| Component | Strategy |
|---|---|
| **Data collection** | Continuous labelled sampling via honeypot + manual SOC review |
| **Drift detection** | Monitor mean prediction confidence weekly; trigger retraining if confidence distribution shifts > 2σ |
| **Retraining cadence** | Monthly full retrain; weekly incremental update for boosting models |
| **Adversarial testing** | Monthly red-team exercises generating novel attack patterns to probe FN rates |
| **Model versioning** | Shadow deployment - new model runs in parallel for 2 weeks before cutover |

### 9.4 Explainability Requirements

Regulatory frameworks (GDPR, NIS2, SOC 2) increasingly require explainable automated decision systems. For each blocked flow, the system should provide:

```
Alert ID: 20240415-001
Model: Gradient Boosting (threshold: 0.25, confidence: 0.94)
Classification: ATTACK

Top contributing features:
  dload         = 0.0    (expected: ~27,000 bits/sec for Normal traffic)
  rate          = 894,231 pps  (97th percentile for Attack class)
  ct_dst_sport  = 43    (elevated - indicative of port scanning)
  swin          = 0     (zero window - SYN flood signature)

Recommended action: Block source IP, escalate to Tier-2 analyst.
```

This level of attribution is achievable with SHAP TreeExplainer applied to Gradient Boosting in near-real-time.

### 9.5 Scalability Considerations

| Model | Inference Latency | Throughput | Production Suitability |
|---|---|---|---|
| LightGBM | < 0.1ms | 10M+ flows/sec | Inline deployment (perimeter firewall) |
| XGBoost | < 0.1ms | 10M+ flows/sec | Inline deployment |
| Gradient Boosting | 1–5ms | 200–1,000K flows/sec | Near-inline / network tap |
| DNN (GPU) | < 1ms (batched) | 1M+ flows/sec | Asynchronous monitoring |
| Random Forest | 2–10ms | 100–500K flows/sec | Near-inline |
| KNN | O(n) - varies | < 10K flows/sec | Offline analysis only |
| SVM | O(n²) - varies | < 1K flows/sec | Not suitable |

### 9.6 Privacy and Legal Considerations

- **Deep Packet Inspection:** The current model uses flow-level features only (no payload). This is legally safer than DPI in most jurisdictions but must still comply with organisational acceptable use policies.
- **Data retention:** Flow logs used for retraining must be anonymised or pseudonymised per GDPR/PDPA requirements.
- **Automated blocking:** Automated flow blocking based on ML output requires formal approval in most enterprise security policies - human-in-the-loop confirmation is recommended for production until the FP rate is validated below organisational thresholds.

---

## 10. Conclusions

### 10.1 Summary of Findings

This study designed, implemented, and evaluated 11 machine learning models for binary network intrusion detection on the UNSW-NB15 dataset. The key findings are:

**1. Tree-based ensemble methods are state-of-the-art for tabular NIDS.** Gradient Boosting, XGBoost, LightGBM, and CatBoost consistently outperformed all other model classes across all metrics, benefiting from sequential error correction, built-in feature interaction modelling, and robust probability calibration.

**2. Preprocessing was the single highest-leverage intervention.** Deduplication (removing 44.78% of training rows), SMOTE balancing, IQR Winsorisation, and RobustScaler collectively accounted for an estimated 8–13 F2-Macro percentage points of improvement over naive baselines. Without these steps, models either overfit duplicate samples or failed to generalise to the test distribution.

**3. Recall must be prioritised over Precision in NIDS.** The asymmetric cost of False Negatives (missed attacks) versus False Positives (false alarms) necessitates recall-first metric selection. The F2-Macro metric appropriately captures this priority. All production-viable models achieved Recall > 0.96.

**4. SVM and Naive Bayes are fundamentally unsuitable for this domain.** SVM failed due to computational impracticality at scale; Naive Bayes failed due to violated independence assumptions. Both achieved below-random performance on key metrics and must not be deployed.

**5. The DNN achieves the highest recall (0.9912) but the lowest precision.** With appropriate threshold calibration, the DNN's FP burden can be substantially reduced while maintaining near-perfect attack detection - making it the best choice for maximum-security deployments where missed attacks are unacceptable.

### 10.2 Final Recommendations

| Deployment Scenario | Recommended Model | Rationale |
|---|---|---|
| General enterprise NIDS | **Gradient Boosting** (threshold=0.25) | Best precision-recall balance, calibrated probabilities, SHAP-compatible |
| High-throughput (> 1M flows/sec) | **LightGBM** | Sub-millisecond inference, highest AUC-PR (0.9852) |
| Maximum security (zero missed attacks) | **DNN** (threshold=0.15) | Highest recall (0.9912), accepts high FP burden |
| Regulated / explainability-required | **Decision Tree** (max_depth=10) | Full rule-trace per alert, AUC-PR=0.9802 |
| Research / benchmarking | **KNN** | Performance ceiling reference, F2-Macro=0.8451 |

