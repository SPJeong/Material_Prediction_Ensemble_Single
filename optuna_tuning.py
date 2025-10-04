##### optuna_tuning.py
import optuna
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestRegressor as RFR # RandomForestRegressor
from xgboost import XGBRegressor as XGB # XGBoost Regressor
from sklearn.svm import SVR # Support Vector Machine
from sklearn.linear_model import Ridge as RIDGE # Ridge Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def tune_with_optuna(model_class, model_name, X_train, y_train, X_val, y_val, n_trials=50, device='cpu'):
    """
    Hyperparameter tuning using Optuna

    Args:
        model_class: (e.g. RFR, XGB, SVR, RIDGE.
        model_name (str): for printing
        X_train, y_train: training data.
        X_val, y_val: validation data for Optuna.
        n_trials (int): num of trials for Optuna running.
        device (str): only for XGBoost.

    Returns:
        sklearn estimator: return optimized parameters.
    """
    print(f"\n[Optuna Tuning] Starting optimization for {model_name}...")

    # 1. Objective Function define
    def objective(trial):
        # Random Forest (RFR)
        if model_class == RFR:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 75, 150),
                'max_depth': trial.suggest_int('max_depth', 5, 30, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_float('max_features', 0.5, 1.0)
            }
            model = RFR(**params, n_jobs=-1, random_state=777)

        # Extreme Gradient Boosting (XGB)
        elif model_class == XGB:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 75, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5)
            }
            model = XGB(**params, tree_method='hist', device=device, n_jobs=-1, random_state=777)

        # Support Vector Machine (SVR)
        elif model_class == SVR:
            params = {
                'C': trial.suggest_float('C', 0.1, 5, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 0.4, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear'])
            }
            model = SVR(**params)

        # Ridge Regression (RIDGE)
        elif model_class == RIDGE:
            params = {
                'alpha': trial.suggest_float('alpha', 0.1, 2.5, log=True)  # alpha for L2
            }
            model = RIDGE(**params, n_jobs=-1)

        else:
            raise ValueError(f"Unknown model class: {model_class}")

        # model train (X_train, y_train)
        model.fit(X_train, y_train)

        # valiation (X_val, y_val)
        y_pred = model.predict(X_val)

        error = mean_absolute_error(y_val, y_pred)
        return error

    # 2. Optimization to minimizie MAE
    study = optuna.create_study(direction='minimize', study_name=f'{model_name}_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[Optuna Tuning] Best MAE: {study.best_value:.4f}")
    print(f"[Optuna Tuning] Best Params: {study.best_params}")

    # return optimized parameters
    if model_class == XGB:
        best_model = model_class(**study.best_params, tree_method='hist', device=device, n_jobs=-1, random_state=777)
    elif model_class == SVR:
        best_model = model_class(**study.best_params)
    else:
        best_model = model_class(**study.best_params, n_jobs=-1, random_state=777)

    return best_model