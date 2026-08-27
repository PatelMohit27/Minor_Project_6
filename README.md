# Minor_Project_6

# QuickCart Warehouse Inventory — Stockout Risk Prediction

Supervised ML classification project (The Unlox Academy) — predicting whether a product, in a given store, on a given day, is **Safe**, **At-Risk**, or **Imminent** of running out of stock before the next supplier delivery arrives.

Mentored by **Girish** at **The Unlox Academy**.

---

## 📌 Problem Statement

QuickCart is a quick-commerce dark-store operator running 12 stores across 6 Indian metros, stocking 60 SKUs sourced from 15 suppliers. For every SKU, at every store, on every day, the inventory team needs to know: will this item run out of stock before the next replenishment?

This is a 3-class classification problem with a natural class imbalance — a rare, high-cost "Imminent" class sitting inside a much larger "everything is fine" population (similar shape to fraud detection or ER triage problems).

| Class | Definition | Share of data |
|---|---|---|
| Safe | Stock cover exceeds replenishment wait by 3+ days | 65.4% |
| At-Risk | Cover is within 3 days of replenishment wait | 24.0% |
| Imminent | Stock will run out (or already has) before next delivery | 10.6% |

---

## 🗂️ Project Structure

```
├── QuickCart_Stockout_Risk_Classification.ipynb   # Main notebook — full pipeline
├── dim_stores.csv                                 # 12 rows — store dimension
├── dim_skus.csv                                   # 60 rows — product dimension
├── dim_suppliers.csv                              # 15 rows — supplier dimension
├── dim_events.csv                                 # 30 rows — festival/promo calendar
├── fact_inventory_daily.csv                       # 21,600 rows — modeling table
├── QuickCart_Stockout_Risk_Project_Spec.pdf        # Original project spec & data dictionary
└── README.md                                      # This file
```

---

## 🧮 Dataset

5 relational tables joined on `store_id`, `sku_id`, `supplier_id`, and `date`:

- **dim_stores.csv** — city, store size, sqft, year opened
- **dim_skus.csv** — category, price, shelf life, perishability, festival relevance, supplier
- **dim_suppliers.csv** — reliability score, base lead time, lead-time variance
- **dim_events.csv** — daily calendar with festival/promo demand multipliers (Diwali week: Oct 22–26, 2026)
- **fact_inventory_daily.csv** — the grain of the model: one row per (store, SKU, date) = 12 × 60 × 30 = **21,600 rows**

Two intentional data-quality issues are identified and handled in the notebook:
1. **City casing mismatch** — `city_display` in `dim_stores.csv` doesn't always match the clean join key `city`.
2. **The `'N/A'` pandas gotcha** — 3 suppliers store `reliability_score` as the literal text `'N/A'`, which `pd.read_csv()` silently converts to `NaN` by default (only because `'N/A'` happens to be in pandas' built-in missing-value list).

---

## 🔧 Approach

1. **Load & validate** — verify row counts and target distribution against the spec's locked sanity numbers.
2. **Join** — merge all 5 tables into a single modeling frame.
3. **EDA** — confirm three key drivers of stockout risk:
   - Festival week more than doubles the Imminent rate (9.5% → 23.3%)
   - Low-reliability suppliers (<0.75) see ~4x the Imminent rate of high-reliability ones (≥0.85)
   - Perishable SKUs run a higher Imminent rate (12.8% vs 9.3%)
4. **Feature engineering** — `reorder_gap`, `days_of_cover_ratio`, cleaned/imputed supplier reliability, `is_recent_reorder` (rolling groupby-shift), temporal features around the Diwali spike.
5. **Train/test split** — **time-based**, not random (train: Oct 1–23, test: Oct 24–30), since a random row split would leak information across the daily panel of repeated store-SKU pairs.
6. **Models compared:**
   - Baseline (majority-class classifier)
   - Multinomial Logistic Regression (with `class_weight='balanced'`)
   - Random Forest (with `class_weight='balanced'`, feature importances)
7. **Evaluation metric focus** — **recall on the Imminent class**, since missing a real stockout is far costlier than a false alarm.

---

## 📊 Key Results

| Model | Imminent Recall | Macro F1 | Overall Accuracy |
|---|---|---|---|
| Baseline (majority class) | ~0.00 | low | ~0.65 |
| Logistic Regression | improved | improved | trade-off vs baseline |
| Random Forest | best | best | trade-off vs baseline |

(Exact numbers are in the notebook — see Section 10, "Model Comparison.")

Top predictive features (Random Forest): `days_of_cover`, `days_of_cover_ratio`, `reorder_gap`, `sales_velocity_7d`, along with festival-week and supplier-reliability flags — consistent with the EDA findings.

---

## Some insights..
<img width="873" height="501" alt="Screenshot 2026-08-27 102037" src="https://github.com/user-attachments/assets/34ebc493-d8b6-4fac-8d7c-3af80052540f" />
<img width="1317" height="677" alt="Screenshot 2026-08-27 102056" src="https://github.com/user-attachments/assets/9e4315b0-e1b8-4caa-ac19-6a6907c46ecf" />
<img width="662" height="533" alt="Screenshot 2026-08-27 102116" src="https://github.com/user-attachments/assets/ca303045-ed8c-45f6-bbad-9807b6aaf1e1" />
<img width="847" height="577" alt="Screenshot 2026-08-27 102153" src="https://github.com/user-attachments/assets/152ba1c6-1d97-470f-8ab0-8ed29bc4198b" />



## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook QuickCart_Stockout_Risk_Classification.ipynb
```

Run all cells top to bottom — the notebook is self-contained and expects the 5 CSV files in the same directory.

---

## 🙏 Acknowledgements

This project was completed as part of a supervised ML classification module at **The Unlox Academy**, under the mentorship of **Girish**. Thanks to The Unlox Academy for designing a realistic, multi-table dataset with genuine data-quality gotchas and real feature-engineering payoff.
