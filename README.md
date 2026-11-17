# Multi-Class Air Quality Classification Using Machine Learning

## 1. Project Title
**Multi-Class Air Quality Classification Using Machine Learning**  
*(การจำแนกระดับความรุนแรงของมลพิษทางอากาศจากหลายปัจจัยด้วยแมชชีนเลิร์นนิง)*

---

## 2. Environmental Problem
Air pollution is a major environmental and public health concern caused by the combined interaction of particulate matter (`PM2.5`, `PM10`), toxic combustion gases (`CO`, `NO2`, `SO2`), meteorological conditions, and proximity to industrial and densely populated urban zones. Relying on a single pollutant is insufficient to capture overall air quality severity. Accurately classifying air quality into actionable severity levels and identifying the primary emission drivers are essential for timely public health warnings and targeted environmental management.

---

## 3. Dataset Source
* **Dataset:** Air Quality and Pollution Assessment Dataset (`updated_pollution_dataset.csv`)
* **Size:** 5,000 samples across 4 air quality severity classes:
  * `Good`: 2,000 samples (40%)
  * `Moderate`: 1,500 samples (30%)
  * `Poor`: 1,000 samples (20%)
  * `Hazardous`: 500 samples (10%)
* **Original Input Features (9):** `Temperature`, `Humidity`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `Proximity_to_Industrial_Areas`, and `Population_Density`.

---

## 4. Research Questions
1. How accurately can a tuned **Gradient Boosting Classifier** predict multi-class air quality severity levels (`Good`, `Moderate`, `Poor`, `Hazardous`) across an imbalanced environmental dataset?
2. Which environmental, meteorological, and spatial parameters serve as the primary drivers pushing air quality into the critical **`Hazardous`** category?

---

## 5. Methods
* **Data Cleaning:** Clipped minor negative sensor artifacts in `PM10` (1 row) and `SO2` (30 rows) to `0` to reflect physical reality.
* **Feature Engineering:** Created 3 domain-specific environmental indicators, expanding the dataset to 12 features:
  * `PM_Ratio = PM2.5 / (PM10 + 1e-5)` (distinguishes combustion aerosols from coarse dust)
  * `Combustion_Index = NO2 + CO` (captures combined fossil-fuel and traffic combustion intensity)
  * `Industrial_Exposure = SO2 / (Proximity_to_Industrial_Areas + 1e-5)` (models industrial plume exposure relative to distance)
* **Data Splitting & Scaling:** Applied an 80/20 stratified train-test split (4,000 train / 1,000 test) and standardized features using `StandardScaler` fitted strictly on the training set to prevent data leakage.
* **Model Training & Tuning:** Trained a baseline `GradientBoostingClassifier` and performed hyperparameter optimization using `GridSearchCV` (5-fold cross-validation, optimizing for `f1_macro`) across `n_estimators`, `learning_rate`, and `max_depth`.
* **Explainable AI (XAI):** Evaluated feature contributions using built-in Gini/Gain Feature Importance and **SHAP (`shap.Explainer`)** summary analysis for the `Hazardous` class.

---

## 6. Main Results
Evaluated on the unseen test set (1,000 samples), the **Tuned Gradient Boosting Classifier** achieved strong overall and class-level performance despite class imbalance:

| Air Quality Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Good** | 0.9926 | 1.0000 | 0.9963 | 400 |
| **Moderate** | 0.9829 | 0.9567 | 0.9696 | 300 |
| **Poor** | 0.8826 | 0.9400 | 0.9104 | 200 |
| **Hazardous** | 0.9239 | 0.8500 | 0.8854 | 100 |
| **Overall / Macro Avg** | **0.9455** | **0.9367** | **0.9404** | **1000** |

* **Overall Test Accuracy:** **96.00%**
* **Macro F1-Score:** **0.9404**

---

## 7. Best Model
* **Selected Algorithm:** **Tuned Gradient Boosting Classifier**
* **Optimal Hyperparameters:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `random_state=42`
* **Key Strength:** Achieves **100% recall** on `Good` air quality and **92.39% precision** on critical `Hazardous` conditions, with misclassifications occurring strictly between adjacent severity tiers (`Poor` vs. `Hazardous`) rather than false-safe predictions.

---

## 8. Feature Importance
Based on Gradient Boosting Feature Importance and SHAP Summary Plot analysis for predicting **`Hazardous`** air quality:
1. **`CO` (Carbon Monoxide):** Strongest positive driver; high `CO` concentrations sharply increase `Hazardous` probability.
2. **`Proximity_to_Industrial_Areas`:** Strongest spatial driver; low distance (close proximity to factories) heavily pushes predictions toward `Hazardous` (SHAP values up to `+0.8`).
3. **`Population_Density`:** High urban density correlates with elevated pollutant accumulation and severe air quality events.
4. **`Combustion_Index` (`NO2 + CO`):** Ranked 4th overall, confirming that combining combustion gases improves severe pollution detection.

---

## 9. Environmental Interpretation
* **Source &
