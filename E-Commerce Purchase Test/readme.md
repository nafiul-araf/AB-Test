# E-Commerce Purchase Value Analysis

## Project Overview

### Problem Statement
In the competitive e-commerce landscape, understanding customer behavior and maximizing revenue is critical for sustainable growth. This project analyzes customer purchase data to understand the relationship between payment methods (credit card vs PayPal) and total purchase value. By leveraging data-driven insights, the goal is to optimize strategies for enhancing revenue while ensuring a positive customer experience.

### Objective
The main objective of this analysis is to explore whether the payment method influences the total purchase value. Using descriptive statistics and hypothesis testing (A/B testing), we aim to determine if customers using credit cards have a different average purchase value compared to those using PayPal. These insights can help e-commerce platforms tailor their payment strategies to improve revenue generation while ensuring customer satisfaction.

### Research Question
Is there a relationship between total purchase value and payment method, and can we influence customer preferences towards payment methods that result in higher purchase values while maintaining a seamless and satisfying shopping experience?

---

## Steps for Data Analysis

### 1. **Importing Dependencies**

The following libraries are imported for performing necessary tasks like data manipulation, visualization, and statistical analysis:

```python
import numpy as np                 # For numerical operations
import pandas as pd                # For data manipulation and analysis
import matplotlib.pyplot as plt    # For data visualization
import seaborn as sns              # For advanced data visualization
import scipy.stats as stats        # For statistical functions
import statsmodels.api as sm       # For statistical modeling and testing
import warnings                    # For handling warnings
```

The `warnings.filterwarnings('ignore')` statement suppresses warnings to ensure a cleaner output.

---

### 2. **Loading the Dataset**

The dataset is loaded into a Pandas DataFrame for analysis:

```python
df = pd.read_csv('https://raw.githubusercontent.com/nafiul-araf/AB-Test/refs/heads/main/E-Commerce%20Purchase%20Test/purchase_data_exe.csv')
```

The first 5 rows of the dataset are displayed to get an initial overview:

```python
df.head()
```

---

### 3. **Exploratory Data Analysis (EDA)**

EDA is conducted to understand the data's basic structure, detect any missing values, and analyze the key variables.

#### a. **Selecting Relevant Columns**
Only relevant columns for the analysis are retained:
```python
df = df[['date', 'payment_method', 'value [USD]', 'time_on_site [Minutes]', 'clicks_in_site']]
```

#### b. **Data Overview**
The dimensions of the dataset and data types of each column are checked:
```python
df.shape
df.dtypes
```

#### c. **Handling Date Column**
The 'date' column is converted to a DateTime object for easier manipulation:
```python
df['date'] = pd.to_datetime(df['date'])
df.dtypes
```

#### d. **Missing Values & Duplicates**
We check for any missing values and duplicated rows:
```python
df.isnull().sum()  # Missing values
df.duplicated().sum()  # Duplicates
```

#### e. **Payment Method Distribution**
The distribution of payment methods (Credit Card vs PayPal) is analyzed:
```python
df.payment_method.value_counts(normalize=True)
```

Key Observations:
- **Credit Card**: 57.83% of transactions
- **PayPal**: 42.16% of transactions
- **Overall**: Credit cards are slightly more popular than PayPal.

#### f. **Descriptive Statistics**
Summary statistics are provided for numeric columns (purchase value, time on site, clicks on site):
```python
df.describe().round(3)
```

---

### 4. **Data Visualizations**

The following visualizations are used to analyze the distribution of purchase values:

#### a. **Purchase Value Distribution (Histogram with KDE)**
```python
sns.histplot(data = df, x = 'value [USD]', kde = True)
plt.title('Distribution of Purchase Amount')
plt.show()
```

#### b. **Purchase Value Distribution (Boxplot)**
```python
sns.boxplot(data = df, y = 'value [USD]')
plt.title('Distribution of Purchase Amount')
plt.show()
```

---

### 5. **Quantile Analysis**

For each numeric column, the following quantile statistics are calculated:
```python
for col in df.select_dtypes(include = np.number).columns:
    print(f"Fpr column: {col}\n")
    print(f"Minimum Value is: {df[col].min()}")
    print(f"1st Quantile is: {df[col].quantile(0.01)}")
    print(f"10th Quantile is: {df[col].quantile(0.10)}")
    print(f"25th Quantile is: {df[col].quantile(0.25)}")
    print(f"50th Quantile is: {df[col].quantile(0.50)}")
    print(f"75th Quantile is: {df[col].quantile(0.75)}")
    print(f"90th Quantile is: {df[col].quantile(0.90)}")
    print(f"95th Quantile is: {df[col].quantile(0.95)}")
    print(f"99th Quantile is: {df[col].quantile(0.99)}")
    print(f"Maximum Value is: {df[col].max()}\n")
```

The quantiles are summarized as follows:

- **Value [USD]**: Most purchases fall between \$57.17 and \$278.82 (Q1 to Q3).
- **Time on Site [Minutes]**: Most customers spend between 13.3 minutes and 43.0 minutes on the site.
- **Clicks in Site**: Most sessions involve between 8 and 19 clicks.

---

### 6. **Outlier Detection and Removal**

Using the IQR method, potential outliers are identified and filtered:

```python
df = df[
    (df['value [USD]'] < 610.30) &
    (df['time_on_site [Minutes]'] < 87.55) &
    (df['clicks_in_site'] < 36)
]
```

After filtering, the quantile statistics are recalculated for the updated dataset.

---

### 7. **Final Dataset**
The index of the DataFrame is reset to a default integer index:

```python
df.reset_index(drop=True, inplace=True)
```


#### 8. **Exploratory Data Analysis (EDA)**
- **Looping through Numeric Columns for Distribution Visualization:**
  ```python
  for col in df.select_dtypes(include = np.number).columns:
      sns.histplot(data = df, x = col, kde = True) 
      plt.title(f'Distribution of {col}')
      plt.show()
  ```
  - This block loops through all numeric columns in the dataset and creates histograms for each column along with a kernel density estimate (KDE). The histograms display the distribution of data for each numeric feature in the dataset.

- **Summary of Histogram:**
  - For each of the three key features (`purchase amount`, `time on site`, and `clicks`), the histograms reveal skewed distributions, where most values lie on the lower side of the scale with fewer occurrences as values increase.

- **Pie Chart of Payment Method Preferences:**
  ```python
  plt.pie(df['payment_method'].value_counts(normalize=True), labels=df['payment_method'].value_counts().index, startangle=90, shadow=True, autopct='%1.1f%%', colors=['red', 'darkblue'])
  plt.show()
  ```
  - This code creates a pie chart to display the proportion of transactions made using Credit Cards versus PayPal. The chart provides insights into the payment method preference distribution, where 58% of transactions were made using credit cards, and 42% using PayPal.

- **Comparison of Purchase Amount, Time on Site, and Clicks by Payment Method:**
```python
fig, ax = plt.subplots(1, 3, figsize = (15, 6))

sns.histplot(data = df, x = 'value [USD]', hue = 'payment_method', palette = {'credit': 'red', 'paypal': 'darkblue'}, ax = ax[0])
ax[0].set_title('Distribution of Purchase Amount by Payment Method')

sns.histplot(data = df, x = 'time_on_site [Minutes]', hue = 'payment_method', palette = {'credit': 'red', 'paypal': 'darkblue'}, ax = ax[1])
ax[1].set_title('Distribution of Time Spent on Site by Payment Method')

sns.histplot(data = df, x = 'clicks_in_site', hue = 'payment_method', palette = {'credit': 'red', 'paypal': 'darkblue'}, ax = ax[2])
ax[2].set_title('Distribution of Purchase Amount by Payment Method')

plt.tight_layout()
plt.show()
```
  - This section compares the distributions of `purchase amount`, `time spent on site`, and `clicks made` for Credit Card and PayPal users. It plots histograms to see how these features vary between the two payment methods.

- **Trends over Time:**
```python
  fig, ax = plt.subplots(1, 3, figsize = (35, 10))

sns.lineplot(data = df, x = 'date', y = 'value [USD]', hue = 'payment_method', palette = {'credit': 'red', 'paypal': 'darkblue'}, ci = None, ax = ax[0])
ax[0].set_title('Trend of Purchase Amount by Payment Method')

sns.lineplot(data = df, x = 'date', y = 'time_on_site [Minutes]', hue = 'payment_method', palette = {'credit': 'red', 'paypal': 'darkblue'}, ci = None, ax = ax[1])
ax[1].set_title('Trend of Time Spent on Site by Payment Method')

sns.lineplot(data = df, x = 'date', y = 'clicks_in_site', hue = 'payment_method', palette = {'credit': 'red', 'paypal': 'darkblue'}, ci = None, ax = ax[2])
ax[2].set_title('Trend of Purchase Amount by Payment Method')

plt.tight_layout()
plt.show()
```
  - Line plots are used to analyze the trends of `purchase amount`, `time on site`, and `clicks` over time for each payment method. These plots help visualize how these metrics fluctuate over the given timeframe for Credit Card and PayPal transactions.

#### 9. **Hypothesis Testing**
- **Shapiro-Wilk Normality Test:**
  ```python
  stat1, p1 = stats.shapiro(data1)
  stat2, p2 = stats.shapiro(data2)
  ```
  - The Shapiro-Wilk test is used to assess the normality of the distributions for the `purchase amount` for both Credit Card and PayPal transactions. The test provides p-values indicating whether the data is normally distributed.

- **Mann-Whitney U Test:**
  ```python
  stat, p = stats.mannwhitneyu(group1, group2)
  ```
  - Given that the data is not normally distributed, the Mann-Whitney U test is used to compare the distributions of purchase amounts between Credit Card and PayPal users. The result helps determine if there is a statistically significant difference between the two payment methods.

#### 10. **Additional Functions**
- **Normality Check Function:**
```python
def normality_check(data1, data2, alpha=0.05, group1='', group2=''):
  stat1, p1 = stats.shapiro(data1)
  stat2, p2 = stats.shapiro(data2)

  # Print test results
  print(f"For {group1}\nTest Statistic: {stat1} and P-Value: {p1}\n")
  print(f"For {group2}\nTest Statistic: {stat2} and P-Value: {p2}\n")

  # Evaluate normality conditions
  if p1 < alpha and p2 < alpha:
    print(f"Both {group1} and {group2} have P < {alpha}. Data in both groups are not normally distributed. Go to Mann-Whitney U-Test.")
  elif p1 > alpha and p2 > alpha:
    print(f"Both {group1} and {group2} have P > {alpha}. Data in both groups are normally distributed. Go to Levene Test to test the homogeneity of variances.")
  elif p1 < alpha and p2 > alpha:
    print(f"{group1} has P < {alpha} (not normally distributed), but {group2} has P > {alpha} (normally distributed). Go to a non-parametric test like Mann-Whitney U-Test.")
  elif p1 > alpha and p2 < alpha:
    print(f"{group1} has P > {alpha} (normally distributed), but {group2} has P < {alpha} (not normally distributed). Go to a non-parametric test like Mann-Whitney U-Test.")
  else:
    print("Unexpected behavior. Check your data and input values.")
```
  - This function performs normality tests for two datasets and provides a recommendation for the subsequent statistical test based on whether the data is normally distributed or not. If both datasets are not normally distributed, it suggests using the Mann-Whitney U-test.

- **Mann-Whitney U Test Function:**
```python
def mann_whitley_U(group1, group2, alpha = 0.05):
  stat, p = stats.mannwhitneyu(group1, group2)

  print(f"Test Statistics: {stat} and P-Value: {p}\n")

  if p < alpha:
    print(f"Given that P < {alpha}, we reject the null hypothesis. There is a significant difference in the distributions of the two groups.")
  else:
    print(f"Given that P > {alpha}, we fail to reject the null hypothesis. There is no significant difference in the distributions of the two groups.")
```
  - This function performs the Mann-Whitney U test and provides the test statistic, p-value, and conclusions about whether the distributions of the two groups are significantly different.

#### Conclusion

1. **Exploratory Data Analysis:**
   - The histograms show that both Credit Card and PayPal payment methods exhibit right-skewed distributions for purchase amounts, time spent on site, and clicks. This suggests that most users engage with the site for shorter durations and with smaller purchase amounts.
   - The pie chart revealed that 58% of transactions are made with Credit Cards and 42% with PayPal, suggesting a slight preference for Credit Cards.
   - Line plots illustrated fluctuations in trends over time, with Credit Card payments showing more dramatic changes in purchase amounts, time spent, and clicks compared to PayPal.

2. **Hypothesis Testing:**
   - **Normality Test:** Both the Credit Card and PayPal purchase amounts did not follow a normal distribution, confirmed by the Shapiro-Wilk test and QQ plots.
   - **Mann-Whitney U Test:** The Mann-Whitney U test revealed that the distributions of purchase amounts for Credit Card and PayPal transactions are significantly different (p-value < 0.05). This suggests that the payment method does influence the distribution of purchase amounts.

3. **Final Conclusion:**
   - Credit Card transactions have a higher mean purchase amount compared to PayPal transactions, with a difference of $8.18. However, both payment methods have a similar distribution shape (right-skewed), with a smaller proportion of transactions involving larger amounts.
   - The significant difference in purchase amounts between the two payment methods may indicate that Credit Card users tend to spend more on average, which could be useful for marketing or product strategies aimed at different customer groups. 
