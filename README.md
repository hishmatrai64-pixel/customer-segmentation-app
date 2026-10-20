# Customer Segmentation (KMeans + Streamlit)

Segments mall customers into 5 groups using **annual income** and **spending score**.

## Method
- Compared KMeans, hierarchical clustering and DBSCAN on 4 features (age, income, spending, gender) and on 2 features (income, spending).
- With 4 features the clusters were mostly split by gender (silhouette: KMeans 27.19).
- Final model: **KMeans on income and spending, 5 clusters, silhouette 55.47**. Age and gender are used only to describe the segments.

| Segment | Suggested action |
|---|---|
| Standard customers | Regular promotions |
| High income, high spending | Premium offers, loyalty programme |
| Young, low income, high spending | Affordable trendy products, bundles |
| High income, low spending | Personalised premium offers |
| Low income, low spending | Low-cost value deals |

## Run the app
```
pip install -r requirements.txt
streamlit run app.py
```
Files needed next to `app.py`: `scaler.joblib`, `kmeans_model.joblib`, `segment_names.joblib` (created by the notebook). Optional: `customer_segmentation data.csv` for the chart.
