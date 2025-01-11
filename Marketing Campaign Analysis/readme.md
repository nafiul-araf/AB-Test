# **Marketing Campaign A/B Testing and Regression Analysis**

### **Problem Statement**  
The analysis investigates the effectiveness of two campaign types—**Ads (control group)** and **PSAs (test group)**—in driving user engagement and conversions. The goal is to determine which campaign type performs better and identify factors influencing ad views and user behavior.

---

### **Purpose**  
To evaluate the trade-offs between **conversion rate** and **ad engagement metrics** across the two groups and explore the key factors influencing ad views. The findings aim to inform strategic decisions regarding campaign optimization.

---

### **Objectives**
1. Assess the differences in **conversion rates** and **ad view rates** between the control and test groups.
2. Identify the key factors driving the total number of ads viewed by users.
3. Build predictive models to estimate ad views based on user behavior and other influencing variables.
4. Provide actionable recommendations aligned with business goals.

---

### **Key Questions**
1. What are the differences in conversion rates and ad view rates between the two groups?
2. What factors influence the total number of ads seen by users?
3. How do user behavior patterns (day, hour, conversion status) affect ad views?
4. Can we predict the total number of ads viewed based on user characteristics and engagement patterns?

---

### **Methods**
1. **Exploratory Data Analysis (EDA):**  
   Visualized the distribution of ads viewed and most active hours across groups.
   
2. **Hypothesis Testing:**  
   - **Z-Test** for conversion rate differences.  
   - **Mann-Whitney U Test** for ads view rate differences.

3. **Regression Analysis:**  
   Built multiple linear regression models to identify significant predictors of ad views and evaluate their impact.

---

### **Insights**
1. **Distribution Analysis:**
   - **Ads Views:** Right-skewed for both groups, indicating most users view fewer ads, with occasional high outliers.
   - **Most Ads Hour:** Multimodal with peaks differing between the groups, with PSA viewers peaking slightly earlier.

2. **Hypothesis Testing Results:**
   - **Conversion Rate:** Ads group had a significantly higher conversion rate (+0.758%) compared to the PSA group.  
   - **Ad View Rate:** PSA group exhibited a slightly higher ad view rate (+0.087).

3. **Regression Analysis Results:**
   - **Key Predictors of Total Ads Viewed:**
     - **Conversion Status:** Users who converted viewed significantly more ads (+51.2 ads).  
     - **Ads Hour and Day:** Statistically significant but minor effects.  
     - **Test Group:** PSA group users viewed slightly more ads (+0.49).
   - Model performance improved with additional predictors but remained moderate (highest R² = 0.0642).
   - Residuals were not normally distributed, suggesting room for model refinement.

---

### **Recommendations**
1. **For Conversion-Focused Campaigns:**
   - Prioritize ad-based campaigns (control group), as they deliver higher conversion rates.  
   - Optimize ad delivery timing based on user behavior patterns.

2. **For Engagement-Focused Campaigns:**
   - Leverage PSA campaigns to increase ad view rates and promote broader awareness.  
   - Investigate factors driving PSA engagement to enhance overall campaign design.

3. **Strategic Decisions:**
   - Conduct a cost-benefit analysis to weigh the financial returns from conversions against broader engagement objectives.
   - Explore hybrid campaigns to balance conversions and engagement.

4. **Future Research:**
   - Refine regression models by including additional predictors (e.g., demographics, user activity data).  
   - Investigate the impact of external factors, such as campaign duration or content type.  
   - Evaluate long-term effects of PSA-driven engagement on brand loyalty.

---

### **Conclusion**  
The analysis reveals a trade-off between conversion rates and engagement metrics. While ad-based campaigns excel in driving conversions, PSA campaigns achieve slightly higher engagement. The findings emphasize aligning campaign strategies with business objectives—whether maximizing immediate conversions or fostering broader user interaction. Statistical models provide insights into user behavior but require further refinement for stronger predictive capabilities.

---

## **Code Explanations**

### **Overview**
This project analyzes an A/B test comparing two marketing strategies: advertisements (control group) and public service announcements (test group). The objective is to evaluate the performance of these groups in driving user conversions, using statistical methods like conversion rate calculations and Z-tests.

---

### **1. Importing Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
```
This section imports essential libraries for data manipulation, visualization, statistical testing, machine learning, and handling warnings. The `ggplot` style is used for consistent visual formatting.

---

### **2. Data Loading and Initial Cleaning**
```python
df = pd.read_csv('/content/drive/MyDrive/A B Test/Marketing Campaign/marketing_AB.csv')
df.drop(['Unnamed: 0', 'user id'], axis=1, inplace=True)
df.head()
```
- The dataset is loaded, and unnecessary columns (`Unnamed: 0` and `user id`) are removed for clarity.
- `head()` displays the first few rows of the data.

---

### **3. Summary Statistics and Data Distribution**
#### **Descriptive Statistics**
```python
df.describe()
```
Provides an overview of numeric columns, including counts, means, standard deviations, and percentiles.

#### **Group Analysis**
```python
df['test group'].value_counts(normalize=True)
```
Shows the proportion of users in each group (`control` vs. `test`), useful for verifying balanced groups.

#### **Histograms of Key Variables**
```python
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(data=df, x='total ads', kde=True, hue='test group', bins=5, ax=ax[0])
ax[0].set_title('Total Ads Views Distribution per Group')

sns.histplot(data=df, x='most ads hour', kde=True, hue='test group', bins=5, ax=ax[1])
ax[1].set_title('Most Ads Hour Distribution per Group')

plt.tight_layout()
plt.show()
```
- Visualizes the distribution of `total ads` and `most ads hour` for each test group.
- Kernel Density Estimation (`kde=True`) smooths the histograms.

---

### **4. Percentile Analysis**
```python
for col in df.select_dtypes(include=np.number).columns:
    print(f"Fpr column: {col}\n") 
    print(f"Minimum Value is: {df[col].min()}")
    # Remaining quantiles and max are printed similarly
```
- Loops through numeric columns to calculate key statistics like percentiles (e.g., 1st, 10th, 25th).

---

### **5. Outlier Removal**
```python
df = df[df['total ads'] <= 276]
df.reset_index(drop=True, inplace=True)
```
Filters out extreme outliers in the `total ads` column based on the 99.5th percentile.

---

### **6. Pie Chart for Categorical Distribution**
```python
colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink']
plt.pie(
    df['most ads day'].value_counts(normalize=True),
    labels=df['most ads day'].value_counts().index,
    startangle=90,
    shadow=True,
    autopct=lambda p: '{:.1f}%'.format(p),
    colors=colors,
    textprops={'color': 'black', 'fontsize': 12}
)
plt.title('Most Ads Views Day')
plt.show()
```
Creates a pie chart for the proportion of ad views by day of the week, highlighting user activity patterns.

---

### **7. A/B Group Splitting**
```python
control_group = df[df['test group'] == 'ad']
test_group = df[df['test group'] == 'psa']
control_group_n, test_group_n = control_group.shape[0], test_group.shape[0]
```
Separates the dataset into control and test groups for comparative analysis.

---

### **8. Conversion Rate Calculations**
```python
control_group_converted = control_group[control_group['converted'] == True]
test_group_converted = test_group[test_group['converted'] == True]
control_group_CR = np.round((control_group_converted_n / control_group_n) * 100, 3)
test_group_CR = np.round((test_group_converted_n / test_group_n) * 100, 3)
difference = np.round(test_group_CR - control_group_CR, 3)
```
- Calculates conversion rates for each group and computes the difference.
- Conversion rates are displayed with contextual interpretations.

---

### **9. Z-Test for Proportions**
#### **Function Definition**
```python
def proportion_ztest(n_converted_control, n_converted_test, n_total_control, n_total_test, alpha=0.05):
    conversions = [n_converted_control, n_converted_test]
    sample_sizes = [n_total_control, n_total_test]
    z_stat, p_value = proportions_ztest(conversions, sample_sizes, alternative='larger')
    
    if p_value < alpha:
        print("Reject the null hypothesis. Control group has a significantly higher conversion rate.")
    else:
        print("Fail to reject the null hypothesis. No significant difference in conversion rates.")
```
This function performs a two-sample Z-test to compare group proportions and interprets the results based on a significance level (`alpha`).

#### **Function Usage**
```python
proportion_ztest(control_group_converted_n, test_group_converted_n, control_group_n, test_group_n)
```
The test determines whether the control group’s conversion rate is statistically significantly higher than the test group’s rate.

---

### Ads View Analysis and Statistical Testing

This section contains Python code for analyzing and testing differences in ad view rates between a control group (users shown ads) and a test group (users shown Public Service Announcements, PSAs). It includes data visualization, normality tests, statistical analysis, and regression modeling.

---

### Calculate Means and Compare Groups
```python
control_group_mean = np.round(control_group['total ads'].mean(), 3)
test_group_mean = np.round(test_group['total ads'].mean(), 3)
difference = np.round(test_group_mean - control_group_mean, 3)

if difference > 0:
  print(f"""Ads View Rate in Control Group (User Saw Ads) is {control_group_mean} and Ads View Rate in Test Group (User Saw Public Service Announcment) is {test_group_mean}. The Ads View Rate in Test Group is {difference} Higher Than the Control Group""")
elif difference < 0:
  print(f"""Ads View Rate in Control Group (User Saw Ads) is {control_group_mean} and Ads View Rate in Test Group (User Saw Public Service Announcment) is {test_group_mean}. The Ads View Rate in Test Group is {np.abs(difference)} Lower Than the Control Group""")
else:
  print(f"""Ads View Rate in Control Group (User Saw Ads) is {control_group_mean} and Ads View Rate in Test Group (User Saw Public Service Announcment) is {test_group_mean}. The Ads View Rate in Test Group is Equal to the Control Group""")
```
This block calculates the means of ad view rates for both groups and compares them. It prints whether the test group performed better, worse, or the same as the control group.

---

### Visualizing Distributions
#### Histogram of Total Ads Viewed
```python
fig, ax = plt.subplots(1, 2, figsize = (15, 6))

sns.histplot(data = control_group, x = 'total ads', kde = True, bins = 5, ax = ax[0])
ax[0].set_title('Total Ads Views Distribution in Control Group')

sns.histplot(data = test_group, x = 'most ads hour', kde = True, bins = 5, ax = ax[1])
ax[1].set_title('Most Ads Hour Distribution Test Group')

plt.tight_layout()
plt.show()
```
This visualization helps identify differences in distributions between the two groups. **Kernel Density Estimation (KDE)** lines show the probability density of the data.

#### QQ Plots for Normality
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sm.qqplot(control_group['total ads'], line='45', ax=axes[0])
axes[0].set_title('Normality Check for Total Ads Views in Control Group')

sm.qqplot(test_group['total ads'], line='45', ax=axes[1])
axes[1].set_title('Normality Check for Total Ads Views in Test Group')

plt.tight_layout()
plt.show()
```
These QQ plots visually check if the distributions of ad views follow a normal distribution.

---

### Normality Check Function
```python
def normality_check(data1, data2, alpha=0.05, group1='', group2=''):
  stat1, p1 = stats.shapiro(data1)
  stat2, p2 = stats.shapiro(data2)

  print(f"For {group1}\nTest Statistic: {stat1} and P-Value: {p1}\n")
  print(f"For {group2}\nTest Statistic: {stat2} and P-Value: {p2}\n")

  if p1 < alpha and p2 < alpha:
    print(f"Both {group1} and {group2} have P < {alpha}. Data in both groups are not normally distributed. Go to Mann-Whitney U-Test.")
  elif p1 > alpha and p2 > alpha:
    print(f"Both {group1} and {group2} have P > {alpha}. Data in both groups are normally distributed. Go to Levene Test to test the homogeneity of variances.")
  elif p1 < alpha or p2 < alpha:
    print(f"One group is not normally distributed. Use a non-parametric test like Mann-Whitney U-Test.")
```
The function conducts **Shapiro-Wilk normality tests** on the data and provides guidance on which statistical test to perform next.

---

### Mann-Whitney U-Test
```python
def mann_whitley_U(group1, group2, alpha = 0.05):
  stat, p = stats.mannwhitneyu(group1, group2)

  print(f"Test Statistics: {stat} and P-Value: {p}\n")

  if p < alpha:
    print(f"Given that P < {alpha}, we reject the null hypothesis. There is a significant difference in the distributions of the two groups.")
  else:
    print(f"Given that P > {alpha}, we fail to reject the null hypothesis. There is no significant difference in the distributions of the two groups.")
```
This block implements the **Mann-Whitney U-test**, a non-parametric test used when normality is not satisfied.

---

### Regression Analysis
#### Regression Function
```python
def regression_analysis(columns, target, test_size):
  X = df[columns]
  y = df[target]

  X = pd.get_dummies(X, drop_first=True).astype(int)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

  X_train_const = sm.add_constant(X_train)
  model = sm.OLS(y_train, X_train_const).fit()
  print(f"\nRegression analysis with the columns: {columns}\n\n{model.summary()}\n")

  X_test_const = sm.add_constant(X_test)
  y_pred = model.predict(X_test_const)

  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  print("\nMean Squared Error (MSE):", mse)
  print("\nR-squared:", r2)

  return X, X_train, X_test, y_train, y_test
```
This function performs **OLS regression**:
1. Encodes categorical variables.
2. Splits the dataset into training and testing sets.
3. Fits the model and calculates metrics like **MSE** and **R-squared**.

#### Single Predictor
```python
X_single, X_train_single, X_test_single, y_train_single, y_test_single = regression_analysis(columns = ['most ads hour'], target = 'total ads', test_size = 0.10)
```

#### Two Predictors
```python
X_double, X_train_double, X_test_double, y_train_double, y_test_double = regression_analysis(columns = ['most ads hour', 'most ads day'], target = 'total ads', test_size = 0.10)
```

#### All Predictors
```python
X_all, X_train_all, X_test_all, y_train_all, y_test_all = regression_analysis(columns = ['most ads hour', 'most ads day', 'converted'], target = 'total ads', test_size = 0.10)
```

#### Including Test Group
```python
df['test group'] = df['test group'].map({'ad': 0, 'psa': 1})

X_group, X_train_group, X_test_group, y_train_group, y_test_group = regression_analysis(columns = ['test group', 'most ads hour', 'most ads day', 'converted'], target = 'total ads', test_size = 0.10)
```
This adds the test group as a variable to the regression, allowing analysis of its impact on the outcome.

---

## **Conclusion**
The analysis reveals a trade-off between conversion rates and engagement metrics. While ad-based campaigns excel in driving conversions, PSA campaigns achieve slightly higher engagement. The findings emphasize aligning campaign strategies with business objectives—whether maximizing immediate conversions or fostering broader user interaction. Statistical models provide insights into user behavior but require further refinement for stronger predictive capabilities.
