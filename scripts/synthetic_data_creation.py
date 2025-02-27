import pandas as pd
import numpy as np
# Set random seed for reproducibility
np.random.seed(42)
# Number of synthetic records to generate
n_records = 10000
# Generate synthetic data
data = {
    # Business Profile
    "business_size": np.random.choice(["small", "medium", "large"], size=n_records, p=[0.6, 0.3, 0.1]),
    "business_time": np.random.randint(1, 20, size=n_records),  # Years in operation
    "business_industry": np.random.choice(["retail", "hospitality", "manufacturing", "services"], size=n_records),
    "business_structure": np.random.choice(["sole trader", "partnership", "company"], size=n_records),
    "business_location": np.random.choice(["urban", "suburban", "rural"], size=n_records),
    "business_review": np.round(np.random.uniform(1, 5, size=n_records), 1),  # Google review score (1-5)
    # Financial Data
    "fin_revenue_t12m": np.random.randint(50000, 1000000, size=n_records),  # Annual revenue
    "fin_profit_margin_t12m": np.round(np.random.uniform(0.05, 0.25, size=n_records), 2),  # Profit margin (5%-25%)
    "fin_liquidity_ratio_t3m": np.round(np.random.uniform(0.5, 3.0, size=n_records), 2),  # Current ratio (0.5-3.0)
    # Credit History
    "credit_score": np.random.randint(300, 1000, size=n_records),  # Credit score (300-1000)
    "credit_accounts_cnt": np.random.randint(1, 10, size=n_records),  # Number of active credit accounts
    "credit_current_amt": np.random.randint(1000, 100000, size=n_records),  # Total outstanding credit amount
    "credit_default_cnt_t12m": np.random.choice([0, 1, 2], size=n_records, p=[0.8, 0.15, 0.05]),  # Defaults in last 12 months
    "credit_inquiry_cnt_t3m": np.random.choice([0, 1, 2, 3], size=n_records, p=[0.6, 0.3, 0.08, 0.02]),  # Credit inquiries in last 3 months
    "credit_court_judgement_cnt": np.random.choice([0, 1, 2], size=n_records, p=[0.8, 0.15, 0.05]),  # Court judgments
    # Transaction Data
    "tran_volume_t1m": np.random.randint(1000, 50000, size=n_records),  # Transaction volume in last 1 month
    "tran_cnt_t1m": np.random.randint(10, 200, size=n_records),  # Number of transactions in last 1 month
    "tran_volume_t3m": np.random.randint(3000, 150000, size=n_records),  # Transaction volume in last 3 months
    "tran_cnt_t3m": np.random.randint(30, 600, size=n_records),  # Number of transactions in last 3 months
    "tran_inflow_avg_t3m": np.random.randint(5000, 50000, size=n_records),  # Average cash inflow in last 3 months
    "tran_outflow_avg_t3m": np.random.randint(4000, 45000, size=n_records),  # Average cash outflow in last 3 months
    # Platform Behavior
    "plat_login_cnt_t1m": np.random.randint(1, 50, size=n_records),  # Logins to Zeller’s platform in last 1 month
    "plat_cust_service_cnt_t3m": np.random.choice([0, 1, 2, 3], size=n_records, p=[0.7, 0.2, 0.08, 0.02]),  # Customer service interactions in last 3 months
    "plat_viztool_usage_cnt_t3m": np.random.choice([0, 1, 2, 3], size=n_records, p=[0.6, 0.3, 0.08, 0.02]),  # Usage of visualization tools in last 3 months
    # Loan Data
    "loan_amount": np.random.randint(10000, 200000, size=n_records),  # Total loan amount outstanding
    "loan_credit_utili_ratio": np.round(np.random.uniform(0.1, 0.9, size=n_records), 2),  # Credit utilization ratio (10%-90%)
    "loan_tenure_remaining": np.random.randint(6, 60, size=n_records),  # Months left in loan term
    "loan_interest_rate": np.round(np.random.uniform(0.05, 0.15, size=n_records), 2),  # Interest rate (5%-15%)
    "default": np.random.choice([0, 1], size=n_records, p=[0.92, 0.08])  # Default status (0 = no, 1 = yes)
}


# Ensure logical consistency between fields
# 1. T3M data should be larger than T1M data
data["tran_volume_t3m"] = np.maximum(data["tran_volume_t3m"], data["tran_volume_t1m"])
data["tran_cnt_t3m"] = np.maximum(data["tran_cnt_t3m"], data["tran_cnt_t1m"])
# 2. Average inflow/outflow should be consistent with transaction volume
data["tran_inflow_avg_t3m"] = np.minimum(data["tran_inflow_avg_t3m"], data["tran_volume_t3m"])
data["tran_outflow_avg_t3m"] = np.minimum(data["tran_outflow_avg_t3m"], data["tran_volume_t3m"])

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data/synthetic_credit_data.csv", index=False)
print("Synthetic data generated and saved to 'data/synthetic_credit_data.csv'.")
