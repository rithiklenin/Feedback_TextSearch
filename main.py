from src.data_loader import load_data
from src.feature_engineering import get_feature_columns, build_preprocessing_pipeline, preprocess_data
from src.model import get_models, train_model, predict
from src.evaluation import evaluate_model

train_path = 'data/train.csv'
test_path = 'data/test.csv'

# Load data
train, test = load_data(train_path, test_path)

# Get numerical and categorical columns
numerical_cols, categorical_cols = get_feature_columns()

# Build the preprocessing pipeline
preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)

# Preprocess the data
X_train, y_train, X_test = preprocess_data(train, test, preprocessor)

# Get models
models = get_models()

# Train and evaluate models
for name, model in models.items():
    trained_model = train_model(X_train, y_train, model)
    predictions = predict(trained_model, X_test)
    print(f"Model: {name}")
    # Since test data does not have y_test, only predict on train for evaluation
    train_preds = predict(trained_model, X_train)
    rmse = evaluate_model(y_train, train_preds)
    print(f"RMSE: {rmse}")
