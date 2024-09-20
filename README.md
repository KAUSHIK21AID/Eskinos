# Eskinos
Welcome to Eskinos, a revolutionary AI tool designed to enhance the diagnosis of Chronic Kidney Disease (CKD) through advanced machine learning and data analysis techniques. Eskinos aims to provide doctors and medical professionals with innovative solutions for better patient care.

Processes Involved:
1. Data Collection

    - Source: Local Multispeciality Hospital
    - Department: Nephrology
    - Data: 305 patients, 4414 datapoints (rows)

2. Data Cleaning, Validation, and Preprocessing

    - Techniques: Imputation, feature selection
    - Methods: Mutual information classification, Random Forest, LightGBM, etc.

3. Module Building

  Predictions:
  Forecaster: For existing patients, using Deep Learning models like S-ARIMA
  Predictor: For new patients, using Regression models like Random Forest Regressor
  
  Recommendation Systems:
  Recommender: For existing patients, utilizing historical medicinal data and current test results
  Suggester: For new patients, comparing first visit results with those of low-risk patients
  
4. Additional Features

  --> Next appointment predictions
  --> Identification of the best medicines
  --> Analysis of frequency vs GFR (Glomerular Filtration Rate) relationships
