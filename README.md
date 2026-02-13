# Customer Segmentation for Marketing Optimization

## Project Overview

This project analyzes credit card customer data (8,500 customers, 18 features, 6-month period) to discover meaningful customer segments using unsupervised machine learning. The segments are designed to help a marketing team personalize campaigns and optimize resource allocation.

## Repository Structure

```
customer-segmentation-ml/
├── data/
│   └── Project1_dataset.csv       # Input dataset (not tracked by git)
├── notebooks/
│   └── Customer_Segmentation_Project.ipynb  # Full analysis notebook
├── reports/
│   └── figures/                   # Generated visualizations (not tracked by git)
├── .gitignore                     # Git ignore rules
├── LICENSE                        # GPLv3+ License
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/customer-segmentation-ml.git
cd customer-segmentation-ml

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/Customer_Segmentation_Project.ipynb
```

Or with JupyterLab:

```bash
jupyter lab notebooks/Customer_Segmentation_Project.ipynb
```

### Dataset

Place the dataset file `Project1_dataset.csv` in the `data/` directory before running the notebook. The dataset is not tracked by git due to its size.

## Key Findings

### Four Customer Segments Identified

| Segment | Customers | Avg Purchases | Avg Balance | Key Behavior |
|---------|-----------|---------------|-------------|--------------|
| **Low Activity / Dormant** | ~2,600 (31%) | Low | Low | Minimal card usage, low engagement |
| **High-Value Active Spender** | ~2,200 (26%) | High | Moderate | Frequent purchases, good payment behavior |
| **Cash Advance Dependent** | ~2,000 (24%) | Low | High | Relies on cash advances, high balance-to-credit ratio |
| **Revolving Balance Carrier** | ~1,700 (20%) | Moderate | High | Carries balances, rarely pays in full |

### Segment Visualization

The PCA projection shows four well-separated clusters in the feature space, with customers naturally grouping along two main axes: **spending behavior** (purchases vs cash advances) and **payment discipline** (full payers vs balance revolvers).

## Methodology

### 1. Exploratory Data Analysis
- Analyzed distributions (most monetary features are heavily right-skewed)
- Correlation analysis revealed key feature relationships
- Scatter plots identified behavioral patterns (purchases vs cash advance usage are inversely related)

### 2. Preprocessing
- Imputed 298 missing `min_payments` values and 1 missing `credit_limit` with median
- Engineered 4 behavioral ratio features: `balance_to_credit`, `purchases_to_payments`, `cash_advance_ratio`, `avg_purchase_size`
- Applied log(1+x) transform to reduce skewness of monetary features
- Standardized all features with StandardScaler

### 3. Dimensionality Reduction
- PCA analysis: 6 components explain 80% of variance, 8 components for 90%
- 2D PCA projection used for visualization

### 4. Clustering
- K-Means with k=2 to k=10 evaluated
- Optimal k=4 selected based on: elbow method (diminishing returns after 4), silhouette analysis, and business interpretability
- Final model: K-Means with k=4, 30 initializations, 500 max iterations

## Marketing Recommendations

| Segment | Strategy | Goal |
|---------|----------|------|
| Low Activity / Dormant | Re-engagement campaigns, spending bonuses | Activate accounts |
| High-Value Active Spender | Premium rewards, credit limit increases, VIP perks | Retention & upsell |
| Cash Advance Dependent | Balance transfer offers, financial wellness tools | Shift to healthier usage |
| Revolving Balance Carrier | Auto-pay promotion, consolidation plans | Manage credit risk |

## Useful Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run notebook from command line (execute all cells and save output)
jupyter nbconvert --to notebook --execute notebooks/Customer_Segmentation_Project.ipynb --output Customer_Segmentation_Project.ipynb

# Export notebook to HTML report
jupyter nbconvert --to html notebooks/Customer_Segmentation_Project.ipynb --output-dir reports/

# Export notebook to PDF
jupyter nbconvert --to pdf notebooks/Customer_Segmentation_Project.ipynb --output-dir reports/

# Run notebook in non-interactive mode
jupyter run notebooks/Customer_Segmentation_Project.ipynb

# Check installed package versions
pip freeze | grep -E "numpy|pandas|matplotlib|seaborn|scikit-learn"
```

## Challenges & Limitations

1. **Skewed data** required careful preprocessing (log transforms)
2. **K-Means assumes spherical clusters** — DBSCAN or GMM could capture more complex shapes
3. **Cross-sectional snapshot** — temporal trends not captured
4. **No single "correct" k** — business interpretability drove final selection
5. **Missing values** imputed with simple median approach

## Tools & Libraries

Python 3, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn (KMeans, PCA, StandardScaler, silhouette_score)

## Authors

Mohammadreza Hendiani

---

## License

This project is licensed under **GNU General Public License v3.0 or later (GPLv3+)**. See [LICENSE](LICENSE) for details.
