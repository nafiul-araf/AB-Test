### **Scenario**

A fast-food chain is introducing a new menu item but is unsure which of three proposed marketing campaigns will maximize its sales. To determine the most effective strategy, the chain has conducted an experiment. Each campaign (Promotion 1, Promotion 2, and Promotion 3) was tested in randomly selected locations across various markets for four weeks. Weekly sales data for the new item was collected during this period.

The **current average weekly sales** for newly launched products is **\$50,000 per location**. The marketing team has set a goal to increase sales by at least **\$4,000 per week per location**. Therefore, any promotion that raises the average weekly sales to **\$54,000 or more** will be considered a success. The experiment aims to identify which, if any, of the campaigns meets or exceeds this benchmark and determines which campaign performs best overall.

--- 

### **Goal**
The goal is to analyze the weekly sales data to evaluate the performance of the three promotional strategies. Using A/B testing techniques, you will determine:
1. Which promotion, if any, significantly increases weekly sales to the target of **$54,000 or more per location**.
2. Whether there are statistically significant differences in the effectiveness of the three campaigns.
3. The marketing campaign that works best for maximizing sales of the new menu item.

--- 

### **Overview**
The code performs an A/B testing analysis to evaluate three marketing campaigns (Promotions 1, 2, and 3) for their impact on weekly sales. It follows a structured data science approach involving:
1. **Data Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Hypothesis Testing** (Normality checks, Kruskal-Wallis Test, and Tukey HSD Test)

---

### **Code Explanation**
#### **Libraries and Data Loading**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import warnings
warnings.filterwarnings('ignore')
```
- Essential libraries for data manipulation (`pandas`), numerical computations (`numpy`), and visualization (`matplotlib`, `seaborn`).
- Statistical testing modules (`scipy`, `statsmodels`) are included to perform hypothesis testing.

#### **Load Data**
```python
df = pd.read_csv('/content/drive/MyDrive/A B Test/WA_Marketing-Campaign.csv')
df.head()
```
- The dataset is loaded into a DataFrame.
- **`df.head()`** displays the first 5 rows to ensure the data loads correctly.

---

### **Data Preprocessing**
1. **Inspect Dataset**
   ```python
   df.info()
   ```
   - Checks column data types and identifies missing values.
   
2. **Convert Columns to Categorical**
   ```python
   cat_cols = ['MarketID', 'LocationID', 'Promotion', 'week']
   for column in cat_cols:
       df[column] = df[column].astype('category')
   ```
   - Columns like `MarketID`, `LocationID`, `Promotion`, and `week` are categorical variables. Converting them ensures better data representation for analysis.

3. **Duplicate Check**
   ```python
   df.duplicated().sum()
   ```
   - Confirms no duplicate rows exist.

---

### **Exploratory Data Analysis (EDA)**
1. **Descriptive Statistics**
   ```python
   df.describe().round(2)
   ```
   - Summarizes data distributions for numeric columns like `AgeOfStore` and `SalesInThousands`.

2. **Visualize Distributions**
   ```python
   sns.histplot(...)
   ```
   - Creates histograms for the distributions of `AgeOfStore` and `SalesInThousands`, segmented by market size.

3. **Average Sales Analysis**
   ```python
   avg_sales_per_promo = df.groupby('Promotion')['SalesInThousands'].mean().reset_index()
   ```
   - Aggregates average weekly sales for each promotion to identify trends.

---

### **Hypothesis Testing**
#### **1. Normality Check**
```python
stats.shapiro(data)
```
- **Shapiro-Wilk Test**: Determines whether sales data follow a normal distribution.
- **Result**: All p-values are less than 0.05, so the sales data are not normally distributed. Non-parametric tests are needed.

#### **2. Kruskal-Wallis Test**
```python
stats.kruskal(group1, group2, group3)
```
- A **non-parametric test** compares the medians of sales data for the three promotions.
- **Result**: Significant differences exist among the promotions.

#### **3. Tukey’s HSD Test**
```python
pairwise_tukeyhsd(endog=df['SalesInThousands'], groups=df['Promotion'], alpha=0.05)
```
- Post-hoc test identifies which specific promotion pairs have significant differences in their average weekly sales.
- **Result**:
  - Promotion 1 significantly outperforms Promotion 2.
  - Promotion 1 and Promotion 3 show slight differences.
  - Promotion 2 has significantly lower sales than the others.

---

### **Effect Size and Sample Size**
1. **Cohen’s d**
   ```python
   (mean_sales - baseline) / std_dev
   ```
   - Measures the effect size (how impactful a promotion is compared to baseline sales).

2. **Sample Size Calculation**
   ```python
   NormalIndPower().solve_power(effect_size, power, alpha)
   ```
   - Determines the minimum required sample size for statistical significance.

---

### **Conclusion**
The code follows a systematic approach:
- **Preprocesses and cleans data**.
- **Analyzes data distributions and trends**.
- Conducts appropriate **statistical tests** to evaluate the promotions’ performance.
- Provides actionable insights into the most effective campaign (Promotion 1).
