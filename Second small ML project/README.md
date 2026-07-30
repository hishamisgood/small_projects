# 🎬 DVD Rental Duration Prediction

An end-to-end Machine Learning regression pipeline built in Python to predict the number of days a customer will keep a rented DVD. This project helps DVD rental companies optimize inventory planning, manage stock availability, and improve operational efficiency.

---

## 📌 Project Overview & Objective

The primary objective of this project is to develop a predictive regression model that estimates DVD rental duration (`rental_length_days`) based on movie characteristics, pricing rates, and special features.

- **Business Target:** Achieve a **Mean Squared Error (MSE) of 3.0 or less** on unseen test data.
- **Model Result:** The final **Random Forest Regressor** achieved an **MSE of < 2.0**, successfully exceeding the target requirement.

---

## 📊 Dataset Description

The dataset (`rental_info.csv`) contains historical DVD rental transaction records with the following features:

| Feature Name | Description |
| :--- | :--- |
| `rental_date` | Timestamp when the DVD was rented |
| `return_date` | Timestamp when the DVD was returned |
| `amount` | Amount paid by the customer for renting the DVD |
| `amount_2` | Square of `amount` (polynomial feature) |
| `rental_rate` | Daily rental rate charged for the DVD |
| `rental_rate_2` | Square of `rental_rate` (polynomial feature) |
| `release_year` | Release year of the movie |
| `length` | Runtime of the movie in minutes |
| `length_2` | Square of `length` (polynomial feature) |
| `replacement_cost` | Cost to replace the DVD if lost or damaged |
| `special_features` | Text description of special features on the DVD |
| `NC-17`, `PG`, `PG-13`, `R` | One-hot encoded dummy variables for MPAA ratings |

---

## 🛠️ Data Preprocessing & Feature Engineering

To prepare the dataset for machine learning, the following steps were performed:

1. **Target Calculation:** Calculated the target variable `rental_length_days` by subtracting `rental_date` from `return_date` and extracting the day component using pandas datetime tools:
   $$\text{rental\_length\_days} = \text{return\_date} - \text{rental\_date}$$

2. **Feature Engineering (Dummy Variables):**
   Parsed the text column `special_features` to construct two binary indicator variables:
   - `deleted_scenes`: Takes value `1` if "Deleted Scenes" is present, else `0`.
   - `behind_the_scenes`: Takes value `1` if "Behind the Scenes" is present, else `0`.

3. **Data Leakage & Matrix Cleaning:**
   To prevent target leakage and errors from raw text features, the following columns were excluded from the feature matrix $X$:
   - `rental_date` & `return_date` (Direct target leakage)
   - `rental_length_days` (Target variable)
   - `special_features` (Unstructured text)

4. **Train / Test Split:**
   Divided the data into an **80% training set** and a **20% testing set** using `train_test_split(..., test_size=0.2, random_state=9)`.

---

## 🚀 Machine Learning & Model Evaluation

Multiple regression strategies were evaluated:

- **Decision Tree Regressor:** Provided a quick baseline, but suffered from high variance and risk of overfitting.
- **Random Forest Regressor:** Selected as the final model. By leveraging an ensemble of decision trees built on bootstrap samples and random feature subsets, the Random Forest model dramatically reduced prediction variance and improved generalization.

### Results

```python
Best Model: RandomForestRegressor(n_estimators=100, random_state=134)
Target MSE: ≤ 3.0
Achieved Test MSE: ~1.98 (Less than 2.0)