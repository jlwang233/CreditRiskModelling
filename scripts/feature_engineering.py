import pandas as pd
import numpy as np
import scorecardpy as sc
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scripts.synthetic_data_creation import SyntheticDataCreator


class FeatureEngineering:
    """A comprehensive class for feature engineering using scorecardpy.
    This class handles feature derivation, WoE binning, and feature selection.
    """

    def __init__(self, df, reference_date):
        """Initialize the FeatureEngineering class.
        Args:
            df (pd.DataFrame): The DataFrame containing raw features.
            reference_date (pd.Timestamp): The reference date for recency calculations.
        """
        self.df = df
        self.df_all = df
        self.reference_date = reference_date
        self.derived_features = []
        self.selected_features_iv = []
        self.selected_features_woe= []
        self.selected_features_xgb= []
        self.bins = {}
        self.break_adj = {}
        self.train=pd.DataFrame()
        self.test=pd.DataFrame()

    def derive_features(self):
        """Derive all features based on predefined calculations.

        Returns:
            pd.DataFrame: The DataFrame with derived features.
        """
        # 2.1.3. Average
        # Calculate drv_profit_pm_avg_t12m: Average monthly net profit over the last 12 months
        self.df["drv_profit_pm_avg_t12m"] = self.df["fin_profit"] / 12

        # Calculate drv_income_pm_avg_t3m: Average monthly cash inflow-outflow over the last 3 months
        self.df["drv_income_pm_avg_t3m"] = (
            self.df["tran_inflow_t3m"] - self.df["tran_outflow_t3m"]
        ) / 3

        # Calculate drv_tran_volume_pm_avg_t3m: Average monthly transaction volume over the last 3 months
        self.df["drv_tran_vol_pm_avg_t3m"] = self.df["tran_vol_t3m"] / 3

        # Calculate drv_tran_cnt_pm_avg_t3m: Average monthly transaction count over the last 3 months
        self.df["drv_tran_cnt_pm_avg_t3m"] = self.df["tran_cnt_t3m"] / 3

        # 2.1.4. Ratio
        # Calculate drv_profit_margin_rate: Profit margin rate (profit / revenue)
        self.df["drv_profit_margin_rate"] = (
            self.df["fin_profit"] / self.df["fin_revenue"]
        )

        # Calculate drv_asset_liability_rate: Asset-to-liability ratio (assets / liabilities)
        self.df["drv_asset_liability_rate"] = (
            self.df["fin_assets"] / self.df["fin_liabilities"]
        )

        # Calculate drv_loan_util_rate: Loan utilization rate (loan_utilization_amount / loan_amount)
        self.df["drv_loan_util_rate"] = (
            self.df["loan_utilization_amount"] / self.df["loan_amount"]
        )

        # Calculate drv_inflow_outflow_rate_t3m: Inflow-outflow ratio over the last 3 months
        self.df["drv_inflow_outflow_rate_t3m"] = (
            self.df["tran_inflow_t3m"] / self.df["tran_outflow_t3m"]
        )

        # 2.1.5. Consistency
        # Calculate drv_addr_cons_flag: Flag for address consistency (simulated)
        self.df["drv_addr_cons_flag"] = np.random.choice(
            [0, 1], size=len(self.df), p=[0.1, 0.9]
        )  # 90% consistent, 10% inconsistent
        self.df["drv_addr_cons_flag"] = self.df["drv_addr_cons_flag"].astype("category")

        # Calculate drv_industry_cons_flag: Flag for industry consistency (simulated)
        self.df["drv_industry_cons_flag"] = np.random.choice(
            [0, 1], size=len(self.df), p=[0.1, 0.9]
        )  # 90% consistent, 10% inconsistent
        self.df["drv_industry_cons_flag"] = self.df["drv_industry_cons_flag"].astype(
            "category"
        )

        # 2.1.6. Recency
        # Calculate drv_mon_since_apply: Months since loan application
        self.df["drv_mon_since_apply"] = (
            (self.reference_date - pd.to_datetime(self.df["loan_start_date"]))
            / np.timedelta64(30, "D")
        ).astype(int)

        # Calculate drv_mon_since_last_late: Months since last late repayment
        self.df["drv_mon_since_last_late"] = (
            (self.reference_date - pd.to_datetime(self.df["loan_last_late_date"]))
            / np.timedelta64(30, "D")
        ).astype(int)

        # 2.1.7. Standard Deviation
        # Calculate drv_tran_volume_std_t12m: Standard deviation of transaction volume over the last 12 months (simulated)
        self.df["drv_tran_vol_std_t12m"] = np.random.uniform(0, 1000, size=len(self.df))

        ## if we have real data say tran_vol_m1 to tran_vol_m12, we can calculated by:
        # tran_vol_columns = [f'tran_vol_m{i}' for i in range(1, 13)]
        # self.df['drv_tran_vol_std_t12m'] = self.df[tran_vol_columns].std(axis=1)

        # Calculate drv_inflow_std_t6m: Standard deviation of cash inflow over the last 6 months (simulated)
        self.df["drv_inflow_std_t6m"] = np.random.uniform(0, 1000, size=len(self.df))

        # 2.1.8. Proportion
        # Calculate drv_trans_vol_prop_t1t3: Proportion of transaction volume in the last 1 month relative to the last 3 months
        self.df["drv_trans_vol_prop_t1t3"] = (
            self.df["tran_vol_t1m"] / self.df["tran_vol_t3m"]
        )

        # Calculate drv_credit_inquiry_prop_t3t12: Proportion of credit inquiries in the last 3 months relative to the last 12 months
        self.df["drv_credit_inquiry_prop_t3t12"] = (
            self.df["credit_inquiry_cnt_t3m"] / self.df["credit_inquiry_cnt_t12m"]
        )
        self.df["drv_credit_inquiry_prop_t3t12"] = self.df[
            "drv_credit_inquiry_prop_t3t12"
        ].fillna(0)

        print("Feature Derivation Complete.")
        self.derived_features = self.df.filter(like="drv_").columns
        print("\nFeatures derived are:")
        print(self.derived_features)
        self.df_all=self.df
        
        return self.df

    def train_test_split(self):
        """Split the dataset into train and test sets
        """
        good_set = self.df[self.df["default"] == 0]
        bad_set = self.df[self.df["default"] == 1]

        good_train, good_test = train_test_split(good_set, test_size = 0.3,seed=23)
        bad_train, bad_test = train_test_split(bad_set, test_size = 0.3, seed=23)

        self.train = pd.concat([good_train, bad_train])
        self.test = pd.concat([good_test, bad_test])
        return self.train, self.test
        
    def lr_select_features(self,iv_limit=0.02,var_keep=[]):
        """Select features based on Information Value (IV) using scorecardpy.var_filter.
        Returns:
            list: List of selected feature names.
        """
        self.df = sc.var_filter(self.df, y="default",iv_limit=iv_limit,var_kp=var_keep)
        self.selected_features_iv = self.df.columns
        
        print("IV of remaining features are:")
        print(sc.iv(self.df, "default"))
        
        return self.selected_features_iv

    def lr_transform_features_woe(self, break_adj={}):
        """Perform WoE binning on the specified features using scorecardpy. after initial IV selection
        Args:
            break_adj (dict): Dictionary of manual break adjustments for WoE binning.
        Returns:
            dict: Dictionary of binning results.
        """
        self.break_adj = break_adj
        
        bins = sc.woebin(
            self.df,
            y="default",
            breaks_list=self.break_adj,
            ignore_const_cols=False,
            ignore_datetime_cols=False,
            method="Tree",
        )
        print("Plot binning results...")
        p1 = sc.woebin_plot(bins)
        plt.show(p1)
        self.bins = bins
        
        # woe binning
        self.train_woe = sc.woebin_ply(self.train[self.selected_features_iv], self.bins)
        self.test_woe = sc.woebin_ply(self.test[self.selected_features_iv], self.bins)
        
        return self.train_woe,self.test_woe
    
    def lr_stepwise_selection(self, save_list=[], threshold_in=0.05, threshold_out=0.1):
        """Perform a forward-backward feature selection based on p-value from statsmodels.api.OLS.
        Args:
            save_list (list): List of features to start with.
            threshold_in (float): Include a feature if its p-value < threshold_in.
            threshold_out (float): Exclude a feature if its p-value > threshold_out.
        Returns:
            list: List of selected features.
        """
        X = self.train_woe.loc[:, self.train_woe.columns != "default"]
        y = self.train_woe.loc[:, "default"]
        included = list(save_list)
        while True:
            changed = False
            # Forward step
            excluded = list(set(X.columns) - set(included))
            if not excluded:
                break
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.GLM(
                    y,
                    sm.add_constant(pd.DataFrame(X[included + [new_column]])),
                    family=sm.families.Binomial(),
                ).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            
            print("new_pval:", new_pval)
            print("best_pval:", best_pval)
            print("threshold_in:",threshold_in)

            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                print("Add  {:30} with p-value {:.6}".format(best_feature, best_pval))

            # Backward step
            if not included:
                break
            model = sm.GLM(
                y,
                sm.add_constant(pd.DataFrame(X[included])),
                family=sm.families.Binomial(),
            ).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                changed = True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                print("Drop {:30} with p-value {:.6}".format(worst_feature, worst_pval))
            if not changed:
                break
            
        self.selected_features_woe = included.append('default')
        return self.selected_features_woe



# Example Usage:
if __name__ == "__main__":
    # Step 1: Load your raw dataset
    sdc = SyntheticDataCreator()
    df = sdc.generate_synthetic_data()
    reference_date = sdc.reference_date
    # Step 2: Initialize Feature Engineering
    fe = FeatureEngineering(df, reference_date=reference_date)
    # Step 3: Derive Features
    derived_df = fe.derive_features()
    print("Derived Features:", fe.derived_features)
    # Step 4: Train/Test Data Split
    train,test=fe.train_test_split()
    
    # Step 5 - LR:
    # 5.1.1 Select Features initially based on IV
    selected_features = fe.select_features()
    print("Selected Features:", selected_features)
    # 5.1.2 Perform WoE Binning
    train_woe,test_woe = fe.transform_features_woe()
    # 5.1.3: Perform Stepwise Selection
    final_features = fe.lr_stepwise_selection()

    # Step 5 - XGB
    # 5.2.1 
