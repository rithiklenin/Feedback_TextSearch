from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Function to define numerical and categorical columns
def get_feature_columns():
    numerical_cols = [
        'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
        'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 
        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
        '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
        'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
    ]
    
    categorical_cols = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 
        'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
        'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    return numerical_cols, categorical_cols

# Preprocessing pipeline
def build_preprocessing_pipeline(numerical_cols, categorical_cols):
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    return preprocessor

# Function to preprocess train and test data
def preprocess_data(train, test, preprocessor):
    X_train = train.drop('SalePrice', axis=1)
    y_train = train['SalePrice']
    X_test = test.copy()
    
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, y_train, X_test
