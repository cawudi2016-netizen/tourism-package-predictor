import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

def train_model():
    # 1. Load preprocessed splits
    if not (os.path.exists('Xtrain.csv') and os.path.exists('ytrain.csv')):
        raise FileNotFoundError("Preprocessed data splits (Xtrain.csv, ytrain.csv) not found.")

    X_train = pd.read_csv('Xtrain.csv')
    y_train = pd.read_csv('ytrain.csv').values.ravel()
    X_test = pd.read_csv('Xtest.csv')
    y_test = pd.read_csv('ytest.csv').values.ravel()

    # Identify categorical and numerical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    # 2. Build Pipeline
    preprocessor = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown='ignore'), cat_cols)
    )

    classifier = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    pipeline = make_pipeline(preprocessor, classifier)

    # 3. Hyperparameter tuning grid
    param_grid = {
        'xgbclassifier__n_estimators': [50, 100],
        'xgbclassifier__max_depth': [3, 5, 7],
        'xgbclassifier__learning_rate': [0.05, 0.1, 0.2]
    }

    # MLflow Auto-logging or manual tracking
    mlflow.set_experiment("Tourism_Package_Prediction")
    with mlflow.start_run():
        print("--- Starting Hyperparameter Tuning ---")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # 4. Evaluate Model
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        print("
--- Evaluation Metrics ---")
        print(classification_report(y_test, predictions))

        # Log params and metrics to MLflow
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save and log the pipeline
        os.makedirs("tourism_project/deployment", exist_ok=True)
        model_path = "tourism_project/deployment/model.joblib"
        joblib.dump(best_model, model_path)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"Best model saved successfully to {model_path}")

if __name__ == '__main__':
    train_model()
