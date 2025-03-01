import pandas as pd
import numpy as np
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle

seed = 23


class ModelTraining:
    """A class for training and evaluating models (LR, XGB, ANN)."""

    def __init__(self, fe):
        """Initialize the ModelTraining class.
        Args:
            df (pd.DataFrame): The DataFrame containing features and target.
            target (str): The name of the target column.
        """
        self.fe = fe

        self.modle_lr = None
        self.model_xgb = None
        self.model_ann = None

    def train_test_split(self, models=["LR", "XGB", "ANN"]):
        """Split the data into train and test sets and save them as CSV files for specified models."""
        for model_type in models:
            if model_type == "LR":
                if len(self.fe.df_lr) > 1:
                    self.X_train_lr = self.fe.df_lr.loc[
                        self.fe.train_idx, self.fe.df_lr.columns != "default"
                    ]
                    self.y_train_lr = self.fe.df_lr.loc[self.fe.train_idx, "default"]
                    self.X_test_lr = self.fe.df_lr.loc[
                        self.fe.test_idx, self.fe.df_lr.columns != "default"
                    ]
                    self.y_test_lr = self.fe.df_lr.loc[self.fe.test_idx, "default"]
                    # Save datasets
                    self.X_train_lr.to_csv("data/X_train_lr.csv", index=False)
                    self.y_train_lr.to_csv("data/y_train_lr.csv", index=False)
                    self.X_test_lr.to_csv("data/X_test_lr.csv", index=False)
                    self.y_test_lr.to_csv("data/y_test_lr.csv", index=False)
                else:
                    print("Skipping LR: fe.df_lr has 1 or fewer rows.")
            elif model_type == "XGB":
                if len(self.fe.df_xgb) > 1:
                    self.X_train_xgb = self.fe.df_xgb.loc[
                        self.fe.train_idx, self.fe.df_xgb.columns != "default"
                    ]
                    self.y_train_xgb = self.fe.df_xgb.loc[self.fe.train_idx, "default"]
                    self.X_test_xgb = self.fe.df_xgb.loc[
                        self.fe.test_idx, self.fe.df_xgb.columns != "default"
                    ]
                    self.y_test_xgb = self.fe.df_xgb.loc[self.fe.test_idx, "default"]
                    # Save datasets
                    self.X_train_xgb.to_csv("data/X_train_xgb.csv", index=False)
                    self.y_train_xgb.to_csv("data/y_train_xgb.csv", index=False)
                    self.X_test_xgb.to_csv("data/X_test_xgb.csv", index=False)
                    self.y_test_xgb.to_csv("data/y_test_xgb.csv", index=False)
                else:
                    print("Skipping XGB: fe.df_xgb has 1 or fewer rows.")
            elif model_type == "ANN":
                if len(self.fe.df_ann) > 1:
                    self.X_train_ann = self.fe.df_ann.loc[
                        self.fe.train_idx, self.fe.df_ann.columns != "default"
                    ]
                    self.y_train_ann = self.fe.df_ann.loc[self.fe.train_idx, "default"]
                    self.X_test_ann = self.fe.df_ann.loc[
                        self.fe.test_idx, self.fe.df_ann.columns != "default"
                    ]
                    self.y_test_ann = self.fe.df_ann.loc[self.fe.test_idx, "default"]
                    # Save datasets
                    self.X_train_ann.to_csv("data/X_train_ann.csv", index=False)
                    self.y_train_ann.to_csv("data/y_train_ann.csv", index=False)
                    self.X_test_ann.to_csv("data/X_test_ann.csv", index=False)
                    self.y_test_ann.to_csv("data/y_test_ann.csv", index=False)
                else:
                    print("Skipping ANN: fe.df_ann has 1 or fewer rows.")
            else:
                raise ValueError(f"Invalid model_type: {model_type}. Use 'LR', 'XGB', or 'ANN'.")
        print("Train and test datasets saved to 'data/' directory.")


    def train_model(self, model_type, params):
        """Train a model (XGB, LR, or ANN) with hyperparameter tuning and save the model."""
        # Select the appropriate train/test datasets
        if model_type == "XGB":
            X_train, y_train = self.X_train_xgb, self.y_train_xgb
            X_test, y_test = self.X_test_xgb, self.y_test_xgb
            model = XGBClassifier(objective="binary:logistic", n_jobs=-1, random_state=42)
        elif model_type == "LR":
            X_train, y_train = self.X_train_lr, self.y_train_lr
            X_test, y_test = self.X_test_lr, self.y_test_lr
            model = LogisticRegression(max_iter=1000, n_jobs=-1)
        elif model_type == "ANN":
            X_train, y_train = self.X_train_ann, self.y_train_ann
            X_test, y_test = self.X_test_ann, self.y_test_ann
            model = MLPClassifier(random_state=42)
        else:
            raise ValueError("Invalid model_type. Use 'XGB', 'LR', or 'ANN'.")
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(model, params, scoring="roc_auc", cv=5)
        grid_search.fit(X_train, y_train)
        # Best model
        print(f"Best parameters for {model_type}:", grid_search.best_params_)
        best_model = grid_search.best_estimator_
        # Save the model to class attributes
        if model_type == "LR":
            self.model_lr = best_model
        elif model_type == "XGB":
            self.model_xgb = best_model
        elif model_type == "ANN":
            self.model_ann = best_model
        # Cross-validation results top 10
        cv_results = pd.DataFrame(grid_search.cv_results_)
        print("Cross-Validation Results:")
        print(cv_results[["params", "mean_test_score", "std_test_score"]][:10])
        # Predictions
        train_pred = best_model.predict_proba(X_train)[:, 1]
        test_pred = best_model.predict_proba(X_test)[:, 1]
        # Evaluate performance
        train_perf = sc.perf_eva(y_train, train_pred, title="train", show_plot=False)
        test_perf = sc.perf_eva(y_test, test_pred, title="test", show_plot=False)
        print(f"Train AUC: {train_perf['AUC']:.3f}, Test AUC: {test_perf['AUC']:.3f}")
        print(f"Train KS: {train_perf['KS']:.3f}, Test KS: {test_perf['KS']:.3f}")
        # Save the model using pickle
        model_path = f"models/best_{model_type.lower()}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"{model_type} model saved to '{model_path}'.")
        return best_model


    def plot_performance_evaluation(self, models=["LR", "XGB", "ANN"]):
        """Plot performance evaluation for specified models."""
        for model_type in models:
            if model_type == "LR":
                X_train, y_train = self.X_train_lr, self.y_train_lr
                X_test, y_test = self.X_test_lr, self.y_test_lr
                model = self.model_lr
            elif model_type == "XGB":
                X_train, y_train = self.X_train_xgb, self.y_train_xgb
                X_test, y_test = self.X_test_xgb, self.y_test_xgb
                model = self.model_xgb
            elif model_type == "ANN":
                X_train, y_train = self.X_train_ann, self.y_train_ann
                X_test, y_test = self.X_test_ann, self.y_test_ann
                model = self.model_ann
            else:
                raise ValueError(f"Invalid model_type: {model_type}. Use 'XGB', 'LR', or 'ANN'.")
            # Predictions
            train_pred = model.predict_proba(X_train)[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]
            # Plot performance evaluation
            train_perf = sc.perf_eva(y_train, train_pred, title=f"{model_type} Train", show_plot=True)
            test_perf = sc.perf_eva(y_test, test_pred, title=f"{model_type} Test", show_plot=True)
            # # Save the plot
            # plt.gcf().set_size_inches(10, 6)
            # plt.savefig(f"models/{model_type.lower()}_evaluation.png")
            # plt.close()
            # print(f"Performance evaluation plot saved to 'models/{model_type.lower()}_evaluation.png'.")


    def plot_feature_importance(self, models=["LR", "XGB", "ANN"]):
        """Plot feature importance or IV for specified models."""
        for model_type in models:
            if model_type == "LR":
                # Calculate IV for Logistic Regression
                iv_df = sc.iv(self.fe.df_lr, "default")
                iv_df.plot(kind="bar", title="Information Value (IV) for Logistic Regression")
                y_label="IV Score"
                plt.ylabel(y_label)
                
            elif model_type == "XGB":
                # Feature importance for XGBoost
                feature_importance = pd.Series(
                    self.model_xgb.feature_importances_, index=self.X_train_xgb.columns
                ).sort_values(ascending=False)
                feature_importance.plot(kind="bar", title="Feature Importances for XGBoost")
                y_label="Feature Importance"
                plt.ylabel(y_label)
                
            elif model_type == "ANN":
                # Feature importance for ANN (using coefficients)
                result = permutation_importance(self.model_ann, self.X_train_ann, self.y_train_ann, n_repeats=10, random_state=23)
                feature_importance = pd.Series(result.importances_mean, index=self.X_train_ann.columns).sort_values(ascending=False)
                feature_importance.plot(kind="bar", title="Permutation Importances for ANN")
                y_label="Permutation Importance"
                plt.ylabel(y_label)
            else:
                raise ValueError(f"Invalid model_type: {model_type}. Use 'XGB', 'LR', or 'ANN'.")
            # Save the plot
            plt.savefig(f"models/{model_type.lower()}_{y_label}.png")
            plt.close()
            print(f"Feature importance plot saved to 'models/{model_type.lower()}_{y_label}.png'.")



if __name__ == "__main__":
    # Step 1: Load your dataset
    df = pd.read_csv("processed_data.csv")  # Replace with your processed dataset
    # Step 2: Initialize ModelTraining
    mt = ModelTraining(fe)
    # Step 3: Ask user for model type
    model_type = input("Enter the model type (LR, XGB, or ANN): ").strip().upper()
    # Step 4: Train and evaluate the selected model
    if model_type == "LR":
        print("Training Logistic Regression (LR)...")
        mt.train_lr()
    elif model_type == "XGB":
        print("Training XGBoost (XGB)...")
        mt.train_xgb()
    elif model_type == "ANN":
        print("Training Artificial Neural Network (ANN)...")
        mt.train_ann()
    else:
        print("Invalid model type. Please choose LR, XGB, or ANN.")
