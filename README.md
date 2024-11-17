# Defect-Prediction-of-Products-using-
#Project Overview This project analyzes manufacturing defect rates and implements a machine learning model to predict defects based on various production parameters. The analysis includes visualization of defect patterns across different shifts and days of the week, followed by a Random Forest classification model for defect prediction. Features

Defect rate analysis by day and shift Heat map visualization of defect patterns Machine learning model for defect prediction Train-test split validation approach

Technologies Used

Python 3.x Libraries:

pandas: Data manipulation and analysis seaborn: Statistical data visualization matplotlib: Basic plotting functionality scikit-learn: Machine learning implementation

Install required packages pip install pandas numpy seaborn matplotlib scikit-learn Run the Jupyter notebook or Python script jupyter notebook defect_analysis.ipynb

Project Structure ├── data/ │ └── synthetic_product_defects.csv ├── notebooks/ │ └── defect_analysis.ipynb ├── src/ │ ├── visualization.py │ └── model.py ├── requirements.txt └── README.md

Data Description The dataset includes the following features:

Day of Week Shift Defective (Target variable) [Add other relevant features]

Analysis Components

Defect Rate Visualization

Heat map of defect rates by day and shift Colorized visualization for pattern identification

Machine Learning Model

Random Forest Classifier 80-20 train-test split Feature engineering and selection

CONCLUSION Project Conclusion:
1. **Problem Identification**: We identified the high rates of defective products, which increase costs, decrease customer satisfaction, and risk brand damage. The goal was to predict product defects based on manufacturing conditions using machine learning.

2. **Data Collection and Preparation**: Relevant data was acquired from sources like manufacturing records and quality control reports. We performed extensive data cleaning, including handling missing values, removing outliers, and normalizing features to prepare the dataset for modeling.

3. **Exploratory Data Analysis**: Visualizations were used to understand feature distributions and detect patterns. Heatmaps, scatter plots, and count plots helped reveal correlations, and patterns across shifts, days, and hours with respect to defect rates.

4. **Feature Engineering**: New features were engineered, such as time-based categorizations (e.g., morning, afternoon) and encoded categorical variables, enhancing the predictive power of the model.

5. **Model Training**: A Random Forest classifier was chosen to predict product defects. After splitting the data, the model was trained on the training set to identify relationships between manufacturing conditions and product quality.

6. **Model Evaluation**: The trained model was evaluated on a test set using metrics like accuracy and a confusion matrix. The model achieved satisfactory performance in predicting defective products, as shown by an accuracy score and a balanced ROC curve.

**Final Outcome**: By implementing a machine learning-based defect prediction model, we have demonstrated that proactive defect prediction is feasible, with potential to reduce defects, lower costs, and improve customer satisfaction. Further improvements could involve tuning the model, exploring additional algorithms, or incorporating more manufacturing variables.
