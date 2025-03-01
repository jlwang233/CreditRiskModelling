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
        self.df['plat_customer_support_cnt_t3m'] = np.random.choice([0, 1, 2, 3], size=self.n_records, p=[0.7, 0.2, 0.08, 0.02])  # Customer support interactions in last 3 months
        self.df['plat_dashboard_cnt_t3m'] = np.random.choice([0, 1, 2, 3], size=self.n_records, p=[0.6, 0.3, 0.08, 0.02])  # Dashboard usage in last 3 months
        self.df['plat_terminal_flag'] = np.random.choice([0, 1], size=self.n_records, p=[0.3, 0.7])  # Whether Zeller terminal is owned
        self.df['plat_trsct_acct_flag'] = np.random.choice([0, 1], size=self.n_records, p=[0.2, 0.8])  # Whether Zeller transaction account is owned
        self.df['plat_saving_acct_flag'] = np.random.choice([0, 1], size=self.n_records, p=[0.5, 0.5])  # Whether Zeller savings account is owned
        self.df['plat_login_cnt_t1m'] = np.random.randint(1, 15, size=self.n_records) + \
            self.df['plat_customer_support_cnt_t3m']+ self.df['plat_dashboard_cnt_t3m']+ \
                self.df['plat_trsct_acct_flag']*np.random.randint(1, 5, size=self.n_records)  # Logins to Zeller’s platform in last 1 month

        
    def generate_finance_data(self):
        """Generate finance-related data with realistic relationships."""
        revenue_base = {
            "small": 50000,
            "medium": 300000,
            "large": 1000000
        }
        self.df['fin_revenue'] = self.df['business_size'].map(revenue_base) * np.random.uniform(0.8,1.2,size=self.n_records)  # Total revenue last financial year          
        self.df['fin_profit'] = self.df['fin_revenue'] * np.random.uniform(0.05, 0.3, size=self.n_records) # Calculate profit based on revenue (net profit margin 5% to 30%)       
        self.df['fin_assets'] = self.df['fin_revenue'] * np.random.uniform(2, 10, size=self.n_records)  # 2x to 10x annual revenue
        self.df['fin_liabilities'] = self.df['fin_assets'] * np.random.uniform(0.1, 0.5, size=self.n_records)  # 10% to 50% of assets
        
    def generate_transaction_data(self):
        """Generate transactional data with realistic relationships."""
        # Transaction volume scaled by revenue
        self.df['tran_vol_t1m'] = self.df['fin_revenue'] / 12 * np.random.uniform(0.5, 2, size=self.n_records)  # 0.5x to 2x monthly revenue
        self.df['tran_vol_t3m'] = self.df['tran_vol_t1m'] * np.random.uniform(1, 6, size=self.n_records)  # 1x to 6x of monthly volume
        
        # Transaction counts scaled by business size
        cnt_base = {
            "small": 100,
            "medium": 500,
            "large": 1000
        }
        self.df['tran_cnt_t1m'] = self.df['business_size'].map(cnt_base)* np.random.uniform(0.5,1.5,size=self.n_records).astype(int) 
        self.df['tran_cnt_t3m'] = self.df['tran_cnt_t1m'] * np.random.uniform(1, 6, size=self.n_records)  # 1x to 6x monthly count
        
        # Cash inflow and outflow
        self.df['tran_inflow_t3m'] = self.df['fin_revenue'] / 4 * np.random.uniform(0.8, 1.2, size=self.n_records)  # 80% to 120% of monthly revenue
        self.df['tran_outflow_t3m'] = self.df['tran_inflow_t3m'] * np.random.uniform(0.8, 1.2, size=self.n_records)  # 80% to 120% of inflow
        
    def generate_credit_history(self):
        """Generate credit history data."""
        credit_score_base = 500  # Minimum credit score
        credit_score_cap = 1000  # Range of credit score (300 to 1000)
        
        self.df['credit_score'] = credit_score_base + (self.df['fin_profit'] / self.df['fin_profit'].max()) * (credit_score_cap-credit_score_base) * np.random.uniform(0.8,1.2,size=self.n_records)
        self.df['credit_score'] = self.df['credit_score'].clip(credit_score_base, credit_score_cap)  # Ensure credit score stays within bounds
        
        self.df['credit_lines_cnt'] = np.random.randint(1, 10, size=self.n_records)  # Number of active credit lines
        self.df['credit_current_amt'] = self.df['credit_lines_cnt'] * np.random.randint(1000, 100000, size=self.n_records)  # Total outstanding credit amount
        
        self.df['credit_default_cnt_t12m'] = np.random.choice([0, 1], size=self.n_records, p=[0.95, 0.05])  # Defaults in last 12 months yet still available for loans
        self.df['credit_inquiry_cnt_t3m'] = np.random.choice([0, 1, 2, 3], size=self.n_records, p=[0.6, 0.3, 0.08, 0.02])  # Credit inquiries in last 3 months
        self.df['credit_inquiry_cnt_t12m'] = self.df['credit_inquiry_cnt_t3m'] * np.random.uniform(1, 4, size=self.n_records).astype(int)  # 1x to 4x of 3-month inquiries
        self.df['credit_court_cnt_t12m'] = np.random.choice([0, 1, 2], size=self.n_records, p=[0.8, 0.15, 0.05])  # Court judgments in last 12 months
        
    def generate_loan_data(self):
        """Generate loan-related data with constraints."""
        
        self.df['loan_amount'] = np.random.randint(1000, 50000, size=self.n_records)  # Total loan amount outstanding
        self.df['loan_late_repayment_cnt'] = np.random.choice([0, 1, 2], size=self.n_records, p=[0.8, 0.15, 0.05])  # Late repayments(late but not defaulted)
        self.df['loan_interest_rate'] = np.round(np.random.uniform(0.10, 0.20, size=self.n_records), 2)  # Interest rate applied to the loan

        # Loan start date 
        ## must be before reference_date
        random_months_in_term = np.random.randint(1, 12, size=self.n_records)  # Random number of months (1-11)
        self.df['loan_start_date'] = self.reference_date - pd.to_timedelta(random_months_in_term, unit='m')  
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
        # loan_utilization amount
        self.df['loan_utilization_amount'] = self.df['loan_amount'] * (random_months_in_term/random_months_maturity)* np.round(np.random.uniform(0.5, 1.5, size=self.n_records), 2)  # Proportion of available credit being used
        self.df['loan_utilization_amount'] = np.where(
            self.df['loan_utilization_amount'] > self.df['loan_amount'] ,
            self.df['loan_amount'] ,
            self.df['loan_utilization_amount']
        )
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
        
        # Generate Default
        default_prob = (1 - (self.df['credit_score']-500)/1000) * self.df['loan_interest_rate'] * np.random.uniform(0.8, 1.2, size=self.n_records)
        threshold = np.percentile(default_prob, 80)  # 80th percentile threshold
        self.df['default'] = (default_prob >= threshold).astype(int).astype(str)
        

        
    def compile_data(self):
        """Compile all data into a DataFrame and ensure proper data types."""
        # Ensure dates are in datetime format
        date_columns = [col for col in self.df.columns if col.endswith('date')]
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col]).dt.date
                
        # Round all numeric data to integers except loan_interest_rate
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        numeric_columns = numeric_columns.drop(['default','loan_interest_rate'], errors='ignore')  # Exclude loan_interest_rate
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
        self.compile_data().to_csv("data/synthetic_credit_data.csv", index=False)
        print("\nData saved to /data/synthetic_credit_data.csv")
        
        return self.compile_data()
    
# Example usage:
if __name__ == "__main__":
    generator = SyntheticDataCreator(n_records=10000, reference_date='2024-10-01',seed=23)
    df = generator.generate_synthetic_data()    
    print("Data Generation Complete.")
    print("\nData Preview:")
    print(df.head())
    

