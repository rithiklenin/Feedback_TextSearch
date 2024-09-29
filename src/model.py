from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

# Function to initialize models
def get_models():
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'XGBoost': XGBRegressor(),
        'CatBoost': CatBoostRegressor(verbose=0),
        'LightGBM': lgb.LGBMRegressor()
    }
    return models

# Function to train a model
def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

# Function to predict using a trained model
def predict(model, X_test):
    return model.predict(X_test)
