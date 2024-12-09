# 🌟 Customer segmentation and Forecasting 

<div style="text-align: center;">
  <img src="assets/customer-segmentation.jpg" alt="Project Cover" />
</div>

## 📝 Project Overview

The purpose of this project is to analyse the orders database from Global Superstore, a global vendor for office supplies, furniture and technology, to provide insights that will help maximize profits. 

One of the goals is to perform cluster analysis to identify segments of clients that will allow to better target marketing efforts. The second goal is to use the insights gathered in the analysis to choose a product or sub-category of products to perform a demand forecasting, to ensure supply is fulfilled and minimum excess inventory is accumulated.

---

### 🚀 Features

This project follows a structured approach through the following steps:

1. **Data preliminary exploration and cleaning:**
   - Initial analysis to understand the dataset's structure and variables.
   - Handling null values, duplicates, and inconsistencies.
   - Ensuring data quality for accurate modeling.
2. **EDA:**
   - Analysis at different level of aggregations between orders, products and customers.
   - Temporal analysis to uncover trends and seasonality patterns.
3. **Clustering:**
   - Kmeans as clustering model.
   - Evaluation of different combinations of features, k clusters and dimensionality reduction.
4. **Cluster result analysis:**
   - Analyse and understand characteristics of selected customer segments.
5. **Forecasting**
   - Perform forecasting for demand of high impact products based on customer segment insights.


---

## 📁 Repository Structure

```bash
Clustering-Forecast/
├── assets/                     # Visual assets
├── data/                       # Raw data files
├── notebook/                   # Jupyter notebooks for analysis and modeling
│   ├── 1_exploration_cleaning.ipynb
│   ├── 2_EDA.ipynb
│   ├── 3_clustering_customers.ipynb
│   ├── 4_clustering_customers_v2.ipynb
│   ├── 5_cluster_customer_results_analysis.ipynb
│   ├── 7_forecasting.ipynb
├── notes/                      # Notes and additional documentation
├── src/                        # Python scripts for support functions
│   ├── ab_testing_support.py
│   ├── data_preparation.py
│   ├── data_visualization_support.py
│   ├── soporte_clustering.py
│   ├── soporte_eda.py
│   ├── soporte_outliers.py
│   ├── soporte_preprocesamiento.py
│   ├── soporte_sarima.py
├── .gitignore                  # Git ignore file
├── Pipfile                     # Pipenv configuration file
├── Pipfile.lock                # Locked dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Installation and Requirements

### Python version and libraries used

- Python 3.11+

#### 📚 Libraries Used

This project uses the following Python libraries:

1. **[pandas](https://pandas.pydata.org/docs/)**  
2. **[numpy](https://numpy.org/doc/)**  
3. **[scikit-learn](https://scikit-learn.org/stable/documentation.html)**  
4. **[scipy](https://docs.scipy.org/doc/scipy/)**  
5. **[statsmodels](https://www.statsmodels.org/stable/index.html)**  
6. **[matplotlib](https://matplotlib.org/stable/users/index.html)**  
7. **[seaborn](https://seaborn.pydata.org/)**  
8. **[tqdm](https://tqdm.github.io/)**  
9. **[yellowbrick](https://www.scikit-yb.org/en/latest/)**  
10. **[joblib](https://joblib.readthedocs.io/)**  


### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/username/clustering-forecast
   cd clustering-forecast
   ```

2. Install dependencies:
    After positioning inside the repository folder, run:

   ```bash
   pipenv install
   ```

---

## 🔄 Next Steps

- Add backtesting for complete time series forecasting model validation
- Try forecasting adding discount as an exogenous variable
- Enrich dataset with exogenous variables

---

## 🤝 Contributions

Contributions are welcome! Feel free to fork the repository, create pull requests, or raise issues for discussion.

---

## ✒️ Authors

- **Miguel López Virués** - [GitHub Profile](https://github.com/MiguelLopezVirues)

---

## 📜 License

This project is licensed under the MIT License.

