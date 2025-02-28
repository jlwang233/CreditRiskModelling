import pandas as pd
import numpy as np
from datetime import datetime
import os

class SyntheticDataCreator:
    """A class to generate synthetic financial and transactional data with realistic relationships."""
    
    def __init__(self, n_records=10000, reference_date='2024-10-01', seed=23):
        """Initialize the synthetic data creator with number of records and reference date.
        
        Args:
            n_records (int, optional): Number of records to generate. Defaults to 10000.
            reference_date (str, optional): Reference date for all date-related calculations. Defaults to '2024-10-01'.
        """
        
        self.n_records = n_records
        self.reference_date = pd.to_datetime(reference_date)  # Convert to datetime for consistency
        np.random.seed(seed)  # For reproducibility
        self.df = None
        
    def generate_business_profile(self):
        """Generate business profile data."""
        data = {
            "business_size": np.random.choice(["small", "medium", "large"], size=self.n_records, p=[0.6, 0.3, 0.1]),
            "business_time": np.random.randint(1, 20, size=self.n_records),  # Years in operation (1-20)
            "business_industry": np.random.choice(["retail", "hospitality", "manufacturing", "services"], size=self.n_records),
            "business_structure": np.random.choice(["sole trader", "partnership", "company"], size=self.n_records),
            "business_location": np.random.choice(["urban", "suburban", "rural"], size=self.n_records),
            "business_job_openings": np.random.randint(0, 10, size=self.n_records),  # Job openings (0-10)
            "business_review": np.round(np.random.uniform(1, 5, size=self.n_records), 1),  # Google review score (1-5)
        }
        self.df = pd.DataFrame(data)
        
    def generate_platform_engagement(self):
        """Generate platform engagement data."""
        self.df['plat_login_cnt_t1m'] = np.random.randint(1, 50, size=self.n_records)  # Logins to Zeller’s platform in last 1 month
        self.df['plat_customer_support_cnt_t3m'] = np.random.choice([0, 1, 2, 3], size=self.n_records, p=[0.7, 0.2, 0.08, 0.02])  # Customer support interactions in last 3 months
        self.df['plat_dashboard_cnt_t3m'] = np.random.choice([0, 1, 2, 3], size=self.n_records, p=[0.6, 0.3, 0.08, 0.02])  # Dashboard usage in last 3 months
        self.df['plat_terminal_flag'] = np.random.choice([0, 1], size=self.n_records, p=[0.3, 0.7])  # Whether Zeller terminal is owned
        self.df['plat_trsct_acct_flag'] = np.random.choice([0, 1], size=self.n_records, p=[0.2, 0.8])  # Whether Zeller transaction account is owned
        self.df['plat_saving_acct_flag'] = np.random.choice([0, 1], size=self.n_records, p=[0.5, 0.5])  # Whether Zeller savings account is owned
        
    def generate_finance_data(self):
        """Generate finance-related data with realistic relationships."""
        self.df['fin_revenue'] = np.random.randint(50000, 1000000, size=self.n_records)  # Total revenue last financial year          
        self.df['fin_profit'] = self.df['fin_revenue'] * np.random.uniform(0.05, 0.25, size=self.n_records) # Calculate profit based on revenue (net profit margin 5% to 25%)       
        self.df['fin_assets'] = self.df['fin_revenue'] * np.random.uniform(2, 10, size=self.n_records)  # 2x to 10x annual revenue
        self.df['fin_liabilities'] = self.df['fin_assets'] * np.random.uniform(0.1, 0.5, size=self.n_records)  # 10% to 50% of assets
        
    def generate_transaction_data(self):
        """Generate transactional data with realistic relationships."""
        # Transaction volume scaled by revenue
        self.df['tran_vol_t1m'] = self.df['fin_revenue'] / 12 * np.random.uniform(0.5, 2, size=self.n_records)  # 0.5x to 2x monthly revenue
        self.df['tran_vol_t3m'] = self.df['tran_vol_t1m'] * np.random.uniform(1, 6, size=self.n_records)  # 1x to 6x of monthly volume
        
        # Transaction counts scaled by business size
        self.df['tran_cnt_t1m'] = np.random.randint(10, 200, size=self.n_records)  # 10 to 200 transactions per month
        self.df['tran_cnt_t3m'] = self.df['tran_cnt_t1m'] * np.random.uniform(1, 6, size=self.n_records)  # 1x to 6x monthly count
        
        # Cash inflow and outflow
        self.df['tran_inflow_t3m'] = self.df['fin_revenue'] / 4 * np.random.uniform(0.8, 1.2, size=self.n_records)  # 80% to 120% of monthly revenue
        self.df['tran_outflow_t3m'] = self.df['tran_inflow_t3m'] * np.random.uniform(0.8, 1.2, size=self.n_records)  # 80% to 120% of inflow
        
    def generate_credit_history(self):
        """Generate credit history data."""
        self.df['credit_score'] = np.random.randint(300, 1000, size=self.n_records)  # Business credit score (300-1000)
        self.df['credit_lines_cnt'] = np.random.randint(1, 10, size=self.n_records)  # Number of active credit lines
        self.df['credit_current_amt'] = self.df['credit_lines_cnt'] * np.random.randint(1000, 100000, size=self.n_records)  # Total outstanding credit amount
        
        self.df['credit_default_cnt_t12m'] = np.random.choice([0, 1, 2], size=self.n_records, p=[0.8, 0.15, 0.05])  # Defaults in last 12 months
        self.df['credit_inquiry_cnt_t3m'] = np.random.choice([0, 1, 2, 3], size=self.n_records, p=[0.6, 0.3, 0.08, 0.02])  # Credit inquiries in last 3 months
        self.df['credit_inquiry_cnt_t12m'] = self.df['credit_inquiry_cnt_t3m'] * np.random.uniform(1, 4, size=self.n_records).astype(int)  # 1x to 4x of 3-month inquiries
        self.df['credit_court_cnt_t12m'] = np.random.choice([0, 1, 2], size=self.n_records, p=[0.8, 0.15, 0.05])  # Court judgments in last 12 months
        
    # def generate_loan_data(self):
    #     """Generate loan-related data with constraints."""
    #     reference_date_series = pd.Series([self.reference_date] * self.n_records)

    #     self.df['loan_amount'] = np.random.randint(1000, 50000, size=self.n_records)  # Total loan amount outstanding
    #     self.df['loan_utilization_amount'] = self.df['loan_amount']* np.round(np.random.uniform(0.1, 0.9, size=self.n_records), 2)  # Proportion of available credit being used
    #     self.df['loan_late_repayment_cnt'] = np.random.choice([0, 1, 2], size=self.n_records, p=[0.8, 0.15, 0.05])  # Late repayments within 90 days
        
    #     # Loan start date must be before reference_date
    #     self.df['loan_start_date'] = reference_date_series - pd.DateOffset(months=np.random.randint(1,12 , size=self.n_records))
    #     # Loan start date must be after business start date
    #     self.df['loan_start_date'] = np.where(
    #         self.df['loan_start_date'] < reference_date_series  - pd.DateOffset(years=self.df['business_time']),
    #         reference_date_series  - pd.DateOffset(years=self.df['business_time']),
    #         self.df['loan_start_date']
    #     )
    #     # Loan maturity date after reference_date
    #     self.df['loan_maturity_date'] = reference_date_series  + pd.DateOffset(months=np.random.randint(1, 24, size=self.n_records))
    #     self.df['loan_interest_rate'] = np.round(np.random.uniform(0.05, 0.15, size=self.n_records), 2)  # Interest rate applied to the loan
        
    #     # Generate loan_last_late_date based on late repayment counts
    #     self.df['loan_last_late_date'] = np.where(
    #         self.df['loan_late_repayment_cnt'] > 0,
    #         self.df['loan_start_date'] + pd.DateOffset(days=np.random.randint(1, 365, size=self.n_records)),
    #         pd.NaT  # Null if no late repayments
    #     )
    #     # last_late_Date (if any) should be before last payment date
    #     self.df['loan_last_late_date'] = np.where(
    #         self.df['loan_last_late_date'] > reference_date_series  - pd.DateOffset(months=1),
    #         reference_date_series  - pd.DateOffset(months=1),
    #         self.df['loan_last_late_date']
    #     )
    def generate_loan_data(self):
        """Generate loan-related data with constraints."""
        self.df['loan_amount'] = np.random.randint(1000, 50000, size=self.n_records)  # Total loan amount outstanding
        self.df['loan_utilization_amount'] = self.df['loan_amount']*np.round(np.random.uniform(0.1, 0.9, size=self.n_records), 2)  # Proportion of available credit being used
        self.df['loan_late_repayment_cnt'] = np.random.choice([0, 1, 2], size=self.n_records, p=[0.8, 0.15, 0.05])  # Late repayments(late but not defaulted)
        self.df['loan_interest_rate'] = np.round(np.random.uniform(0.05, 0.15, size=self.n_records), 2)  # Interest rate applied to the loan

        # Loan start date 
        ## must be before reference_date
        random_months_in_term = np.random.randint(1, 12, size=self.n_records)  # Random number of months (1-11)
        self.df['loan_start_date'] = self.reference_date - pd.to_timedelta(random_months_in_term, unit='m')  # Convert months to days
        ## must be after business start date
        business_start_date_series = self.reference_date - pd.to_timedelta(self.df['business_time']*12, unit='m')
        self.df['loan_start_date'] = np.where(
            self.df['loan_start_date'] < business_start_date_series,
            business_start_date_series,
            self.df['loan_start_date']
        )
        
        # Loan maturity date 
        ## must be after reference_date
        random_months_maturity = np.random.choice([12, 24, 36, 48], size=self.n_records, p=[0.25, 0.25, 0.25,0.25]) 
        self.df['loan_maturity_date'] = self.reference_date + pd.to_timedelta(random_months_maturity-random_months_in_term, unit='m')
        
        # loan_last_late_date 
        ## based on late repayment counts
        self.df['loan_last_late_date'] = np.where(
            self.df['loan_late_repayment_cnt'] > 0,
            self.df['loan_start_date'] + pd.to_timedelta(np.random.randint(2, 11, size=self.n_records), unit='m'),
            pd.to_datetime('1900-01-01')  # Default value if no late repayments
        )
        self.df['loan_last_late_date'] = pd.to_datetime(self.df['loan_last_late_date'])

        ## should be before last payment date 
        self.df['loan_last_late_date'] = np.where(
            self.df['loan_last_late_date'] > self.reference_date - pd.to_timedelta(1, unit='m'),
            self.reference_date - pd.to_timedelta(1, unit='m'),
            self.df['loan_last_late_date']
        )

        
    def compile_data(self):
        """Compile all data into a DataFrame and ensure proper data types."""
        # Ensure dates are in datetime format
        date_columns = [col for col in self.df.columns if col.endswith('date')]
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col]).dt.date
                
        # Round all numeric data to integers except loan_interest_rate
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        numeric_columns = numeric_columns.drop('loan_interest_rate', errors='ignore')  # Exclude loan_interest_rate
        self.df[numeric_columns] = self.df[numeric_columns].round().astype(int)

        
        return self.df
        
    def generate_synthetic_data(self):
        """Generate synthetic data with all constraints and relationships."""
        self.generate_business_profile()
        self.generate_platform_engagement()
        self.generate_finance_data()
        self.generate_transaction_data()
        self.generate_credit_history()
        self.generate_loan_data()
        return self.compile_data()
    
# Example usage:
if __name__ == "__main__":
    generator = SyntheticDataCreator(n_records=10000, reference_date='2024-10-01',seed=23)
    df = generator.generate_synthetic_data()    
    print("Data Generation Complete.")
    print("\nData Preview:")
    print(df.head())
    
    os.chdir('/Users/wjl/Documents/GitHub/CreditRiskModelling/')
    df.to_csv("data/synthetic_credit_data.csv", index=False)
    
    print("\nData saved to /data/synthetic_credit_data.csv")
