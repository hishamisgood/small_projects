# 🐧 Unsupervised Species Clustering: Palmer Archipelago Penguins

## 📌 Project Overview
This project applies unsupervised machine learning to group and identify distinct penguin populations from field research data collected in Antarctica. Because species labels were not recorded during data collection, a **$K$-Means Clustering** pipeline was constructed to segment the penguins into 3 clusters based on their morphological traits (bill dimensions, flipper length, body mass, and sex).

---

## 📊 Dataset Information
* **Source:** Data collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER (Long Term Ecological Research Network).
* **File:** `penguins.csv`
* **Features:**
  * `culmen_length_mm`: Culmen length (mm)
  * `culmen_depth_mm`: Culmen depth (mm)
  * `flipper_length_mm`: Flipper length (mm)
  * `body_mass_g`: Body mass (g)
  * `sex`: Penguin sex (`MALE` / `FEMALE`)

---

## ⚙️ Workflow & Architecture

The machine learning workflow follows a strict, leak-free pipeline:

1. **Categorical Preprocessing (One-Hot Encoding):**
   * Converted categorical text data (`sex`) into binary indicators using `pd.get_dummies()` to allow distance-based calculations.
2. **Feature Scaling:**
   * Used `StandardScaler` to force all features onto a uniform scale (mean = 0, standard deviation = 1). This prevents high-magnitude features like `body_mass_g` from dominating Euclidean distance metrics over millimeter scale variables.
3. **Pipeline Construction & Model Fitting:**
   * Built an end-to-end assembly line using Scikit-Learn's `make_pipeline(StandardScaler(), KMeans(n_clusters=3))` to execute feature scaling and cluster optimization seamlessly.
4. **Cluster Assignment & Profiling:**
   * Generated cluster labels via `.predict()`, attached assignments back to the original numerical dataset, and aggregated cluster profiles via `.groupby("labels").mean()`.

---

## 💻 Python Implementation

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Load data
penguins_df = pd.read_csv("penguins.csv")

# 2. Convert categorical features to binary numeric variables
penguins_df = pd.get_dummies(penguins_df)

# 3. Instantiate scaler and KMeans algorithm
norm = StandardScaler()
kmeans = KMeans(n_clusters=3, random_state=42)

# 4. Build and fit pipeline
pipeline = make_pipeline(norm, kmeans)
pipeline.fit(penguins_df)

# 5. Predict cluster labels and append to DataFrame
predict_column = pipeline.predict(penguins_df)
penguins_df["labels"] = predict_column

# 6. Compute mean trait values per cluster
stat_penguins = penguins_df.groupby("labels").mean()
print(stat_penguins)