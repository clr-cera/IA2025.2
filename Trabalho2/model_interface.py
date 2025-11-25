# Class to be used in GUI/script to predict with the trained models
from statsmodels.regression.linear_model import OLSResults
from statsmodels.genmod.generalized_linear_model import GLMResults
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import scipy.stats as ss


def get_gamma_prediction_interval(model, X_new, alpha=0.05, n_simulations=10000):
    """
    Generate prediction intervals for Gamma GLM using simulation.
    """
    # Get fitted values and parameters
    prediction = model.get_prediction(X_new)
    pred_summary = prediction.summary_frame(alpha=alpha)
    mu = pred_summary['mean'].values[0]

    # Get the scale parameter (phi) from the model
    scale = model.scale

    # For Gamma distribution: shape = mu^2 / variance, scale = variance / mu
    # In GLM: variance = phi * mu^2
    shape = mu / scale
    scale_param = scale

    # Simulate predictions
    simulated = np.random.gamma(shape, scale_param, n_simulations)

    # Calculate percentiles
    lower = np.percentile(simulated, 100 * alpha / 2)
    upper = np.percentile(simulated, 100 * (1 - alpha / 2))

    return {
        'mean': mu,
        'mean_ci_lower': pred_summary['mean_ci_lower'].values[0],
        'mean_ci_upper': pred_summary['mean_ci_upper'].values[0],
        'obs_ci_lower': lower,
        'obs_ci_upper': upper
    }

def convert_to_xbg(df : pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns="property_code")

    Y = df["sale_price"]
    # Numerical variables (unaltered)
    num = df.select_dtypes(include = "number").drop(columns=["sale_price", "exp(area_util)", "exp(area_total)"])
    boolean = df.select_dtypes(include = "boolean")
    cat = df.select_dtypes(include = "object").astype("category")
    X = pd.concat([num, boolean, cat], axis = 1)
    return X


class ModelInterface:

    def __init__(self):
        self.ols = OLSResults.load("models/ols.pickle")
        self.glm = GLMResults.load("models/gamma_identity.pickle")
        base_params = {
            'device': 'cpu',  # GPU is hard to configure
            'tree_method': 'hist',
            'enable_categorical': True,
            'random_state': 67,
            'n_jobs': 1,
            'verbosity': 1
        }

        self.xgb = XGBRegressor(**base_params)
        self.xgb.load_model("models/xgb_model.json")
        #self.std_data = pd.read_csv("data/std_data.csv")
        self.full_data = pd.read_csv("data/clean_data_sell.csv")

    # Use a dictionary or pandas row as input, with all the columns/fields used in the model (modelling.ipynb)
    def standardize_record(self, record: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same standardization process to a new record.

        Parameters:
        -----------
        record : dict
            A dictionary containing the features of a new record
        self.full_data : pd.DataFrame
            The original training data used to compute statistics

        Returns:
        --------
        dict : Standardized record ready for prediction
        """
        # Create a copy to avoid modifying the original
        std_record : pd.Series = record.copy()

        # Helper function for standardization
        stdize = lambda series, lim: (series - ss.tmean(series, limits=(None, lim))) / ss.tstd(series, limits=(None, lim))

        # Get numeric columns (excluding the special ones)
        num_cols = self.full_data.select_dtypes(include="number").drop(
            columns=["sale_price", "area_util", "area_total", "condominium_fee"]
        ).columns

        # Standardize area_util
        if "area_util" in std_record.columns:
            mean_val = ss.tmean(self.full_data["area_util"], limits=(None, 1000))
            std_val = ss.tstd(self.full_data["area_util"], limits=(None, 1000))
            std_record["area_util"] = (std_record["area_util"] - mean_val) / std_val

        # Standardize area_total
        if "area_total" in std_record.columns:
            mean_val = ss.tmean(self.full_data["area_total"], limits=(None, 1000))
            std_val = ss.tstd(self.full_data["area_total"], limits=(None, 1000))
            std_record["area_total"] = (std_record["area_total"] - mean_val) / std_val

        # Standardize condominium_fee
        if "condominium_fee" in std_record.columns:
            mean_val = ss.tmean(self.full_data["condominium_fee"], limits=(None, 1700))
            std_val = ss.tstd(self.full_data["condominium_fee"], limits=(None, 1700))
            std_record["condominium_fee"] = (std_record["condominium_fee"] - mean_val) / std_val

        # Create exponential features
        if "area_util" in std_record.columns:
            std_record["exp(area_util)"] = np.exp(std_record["area_util"])

        if "area_total" in std_record.columns:
            std_record["exp(area_total)"] = np.exp(std_record["area_total"])


        # Standardize remaining numeric columns
        for col in num_cols:
            if col in std_record.columns:
                limit = ss.quantile(self.full_data[col], 0.80)
                mean_val = ss.tmean(self.full_data[col], limits=(None, limit))
                std_val = ss.tstd(self.full_data[col], limits=(None, limit))
                std_record[col] = (std_record[col] - mean_val) / std_val

        return std_record

    # ("sale_price ~ bedrooms + bathrooms + parking_spaces + area_total + C(property_type) + "
    #  "condominium_fee + has_pool + has_bbq + has_playground +has_sauna + has_party_room + has_sports_court + "
    #  "has_24h_security + has_laundry + has_closet + has_office + has_pantry + amenity_score")
    def get_predictions(self, record : pd.DataFrame, alpha = 0.05):
        std_record = self.standardize_record(record)
        print(dict(std_record.iloc[0]))
        # Returns DataFrame with columns: mean, mean_se, mean_ci_lower, mean_ci_upper,
        #                                  obs_ci_lower, obs_ci_upper
        ols_pred = self.ols.get_prediction(std_record).summary_frame(alpha=alpha)

        glm_pred = get_gamma_prediction_interval(self.glm, record, alpha)

        xgb_pred = self.xgb.predict(convert_to_xbg(std_record))
        return {
            "ols" : ols_pred,
            "glm" : glm_pred,
            "xgb" : xgb_pred
        }



if __name__ == "__main__" :
    record = pd.DataFrame({'property_code': '84210-S',
 'property_type': 'Casa',
 'property_subtype': 'Padrão',
 'sale_price': np.float64(800000.0),
 'bedrooms': np.int64(2),
 'bathrooms': np.int64(1),
 'parking_spaces': np.int64(2),
 'area_util': np.float64(100.0),
 'area_total': np.float64(242.0),
 'condominium_fee': np.float64(0.0),
 'has_pool': np.True_,
 'has_bbq': np.False_,
 'has_playground': np.False_,
 'has_sauna': np.False_,
 'has_party_room': np.False_,
 'has_sports_court': np.False_,
 'has_24h_security': np.False_,
 'has_laundry': np.True_,
 'has_closet': np.False_,
 'has_office': np.False_,
 'has_pantry': np.False_,
 'size_category': 'large',
 'amenity_score': np.int64(1)}, index=[0])
    api = ModelInterface()
    api.get_predictions(record)