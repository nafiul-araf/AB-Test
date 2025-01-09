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
