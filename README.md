# CreditRiskModelling
#### This repository contains the solution to the Z***** Credit Data Analyst technical task. It includes synthetic data generation, feature engineering, model training, and evaluation of the models.
---
## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Conclusion](#conclusion)
---
## **Project Overview**
The task involves:
1. **Synthetic Data Creation**: Generating a dataset with 10,000 rows and 10+ features relevant to credit risk assessment.
2. **Feature Engineering**: Transforming and aggregating features to improve model performance.
3. **Machine Learning Model Development**: Building and evaluating three credit risk models.
4. **Insights and Conclusion**: Interpreting model results and providing actionable insights.
---
## **Repository Structure**
```
CreditRiskModelling/
├── data/                   # Contains synthetic dataset
│   └── synthetic_credit_data.csv
│   └── train and test data
├── notebooks/              # Jupyter notebooks for analysis
│   └── credit_risk_modeling.ipynb
├── scripts/                # Python scripts for data generation and modeling
│   └── synthetic_data_creation.py
│   └── feature_engineering.py
│   └── model_training.py
├── reports/                # Assessment report
│   └── Credit Data Analysis Final Report.pdf
├── models/                 # Model Files and evaluation metrics visualization
│   └── best_lr_model.pkl
│   └── best_xgb_model.pkl
│   └── best_xgb_model.pkl
│   └── feature_importance.png
├── README.md               # This file
└── requirements.txt        # Python dependencies
```
## Installation  
To get started with this project, follow these steps to set up your environment:  
1. **Clone the Repository**:  
   ```bash  
   git clone https://github.com/jlwang233/CreditRiskModelling.git  
   cd CreditRiskModelling  
   ```  
2. **Set Up a Virtual Environment (Optional but Recommended)**:  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`  
   ```  
3. **Install Required Dependencies**:  
   ```bash  
   pip install -r requirements.txt  
   ```  
4. **Download the Synthetic Dataset**:  
   The synthetic dataset is included in the repository under `data/synthetic_credit_data.csv`.  
5. **Run the Scripts**:  
   - **Data Creation**:  
     ```bash  
     python scripts/synthetic_data_creation.py  
     ```  
   - **Feature Engineering**:  
     ```bash  
     python scripts/feature_engineering.py  
     ```  
   - **Model Training and Evaluation**:  
     ```bash  
     python scripts/model_training.py  
     ```  
6. **Explore the Jupyter Notebook**:  
   Open the Jupyter notebook `notebooks/credt_risk_modelling.ipynb` to see a step-by-step walkthrough of the process.  
## Usage  
This repository is designed to simulate a real-world credit risk modeling workflow. Here’s how to use it:  
- **Data Creation**: Generates a synthetic dataset for modeling.  
- **Feature Engineering**: Prepares the dataset by creating relevant features.  
- **Model Training and Evaluation**: Trains and evaluates credit risk models, saving the best model and performance metrics.  
- **Jupyter Notebook**: Provides an interactive example of how to use the scripts.  
## Methodology  
The project follows a structured approach to credit risk modeling:  
1. **Data Preparation**: Synthetic data is generated to mimic real-world credit data.  
2. **Feature Engineering**: Key features are engineered to improve model performance.  
3. **Modeling**: Multiple models are trained and evaluated using metrics like AUC, KS, and accuracy.  
4. **Optimization**: The best-performing model is selected and saved for reproducibility.  
## Results  
The performance metrics for the models are summarized below:  
| Model | Train AUC | Test AUC | Train KS | Test KS |  
|-------|-----------|----------|----------|---------|  
| LR    | 0.8432    | 0.8435   | 0.5414   | 0.5392  |  
| ANN   | 0.8787    | 0.8584   | 0.6018   | 0.5617  |  
| XGB   | 0.8859    | 0.8621   | 0.6141   | 0.5788  |  
### Key Insights:  
- **XGBoost** achieved the highest Train AUC (0.8859) and Test AUC (0.8621), indicating strong predictive performance.  
- **ANN** also performed well, with a Train AUC of 0.8787 and Test AUC of 0.8584.  
- **LR** showed consistent performance between Train and Test AUC, suggesting minimal overfitting.  
For detailed visualizations of these metrics, refer to the `models/` directory.  

## Conclusion  
This project demonstrates a robust approach to credit risk modeling, leveraging synthetic data and advanced modeling techniques. The repository is designed to be reproducible and well-documented, showcasing my skills in data analytics and credit risk modeling.  
Feel free to explore the repository and reach out if you have any questions or feedback!  
