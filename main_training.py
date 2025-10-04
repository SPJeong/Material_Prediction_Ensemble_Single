##### main_training.py (with Optuna)
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import sklearn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestRegressor as RFR  # RandomForestRegressor
from xgboost import XGBRegressor as XGB  # XGBoost Regressor
from sklearn.svm import SVR  # Support Vector Machine
from sklearn import preprocessing  # mostly for Support Vector Machine
from sklearn.linear_model import Ridge as RIDGE  # Ridge Regression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import CONFIG  # custom.py
import chemical_feature_extraction  # custom.py
import data_extraction  # custom.py
import optuna_tuning

# parameter setting
filtered_num = CONFIG.filtered_num
random_pick_num = CONFIG.random_pick_num
ecfp_radius = CONFIG.ECFP_radius
ecfp_nbits = CONFIG.ECFP_nBits
data_extraction_folder = CONFIG.data_extraction_folder
chemical_feature_extraction_folder = CONFIG.chemical_feature_extraction_folder
plot_save_folder = CONFIG.plot_save_folder
model_save_folder = CONFIG.model_save_folder
os.makedirs(chemical_feature_extraction_folder, exist_ok=True)
os.makedirs(plot_save_folder, exist_ok=True)
os.makedirs(model_save_folder, exist_ok=True)
device = CONFIG.device
optuna_num_trials = CONFIG.optuna_num_trials

Y_total_list = ['Cp', 'Tg', 'Tm', 'Td', 'LOI', ]
# 'YM', 'TSy', 'TSb', 'epsb', 'CED',
# 'Egc', 'Egb', 'Eib', 'Ei', 'Eea', 'nc', 'ne',
# 'permH2', 'permHe', 'permCH4', 'permCO2', 'permN2', 'permO2',
# 'Eat', 'rho', 'Xc', 'Xe']


# model selection
model_name_list = ['Random Forest (RF)', 'Extreme Gradient Boosting (XGB)', 'Support Vector Machine (SVM)',
                   'Ridge Regression (RR)']
save_model_name_list = ['RF', 'XGB', 'SVM', 'RR']
modeling = [RFR, XGB, SVR, RIDGE]

print(
    '--------------------------------------------------------------------------------------------------------------------------')
print('Select Model Number for TRAINING')
print(
    '| 0: Random Forest (RF) | 1: Extreme Gradient Boosting (XGB) | 2: Support Vector Machine (SVM) | 3: Ridge Regression (RR) |')

m = int(input())
model_name = save_model_name_list[m]

initial_model = modeling[m]  # for Optuna tuning
print('The ' + model_name + ' model has been selected!')
print('-----------------------------------------------------')
print('\nrunning on ', model_name)

# load file
file_folder = chemical_feature_extraction_folder
file_name = f'chemical_feature_extraction_len_{filtered_num}_num_{random_pick_num}_scaled_True_ECFP_True_desc_True.csv'
file_raw_path = os.path.join(file_folder, file_name)

if os.path.exists(file_raw_path):
    print(f"Loading existing file from: {file_raw_path}")
    file_raw = pd.read_csv(file_raw_path)

else:
    print(f"File not found. Generating data and saving to: {file_raw_path}")
    file_raw = chemical_feature_extraction.run_feature_extraction(filtered_num=filtered_num,
                                                                  random_pick_num=random_pick_num,
                                                                  data_extraction_folder=data_extraction_folder,
                                                                  ecfp=True,
                                                                  descriptors=True,
                                                                  scale_descriptors=True,
                                                                  ecfp_radius=ecfp_radius,
                                                                  ecfp_nbits=ecfp_nbits,
                                                                  chemical_feature_extraction_folder=chemical_feature_extraction_folder,
                                                                  inference_mode=False,
                                                                  new_smiles_list=None)

# ECFP+ descriptors for X
start_column_index = file_raw.columns.get_loc('0')
end_column_index = file_raw.columns.get_loc('CalcNumBridgeheadAtoms')
X_file = file_raw.iloc[:, start_column_index:end_column_index + 1].copy().to_numpy().astype('float32')

# total targets for Y
start_column_index = file_raw.columns.get_loc('Egc')
end_column_index = file_raw.columns.get_loc('Tm')
Y_total_file = file_raw.iloc[:, start_column_index:end_column_index + 1]

print("\n Start training...")
for i, target_name in tqdm(enumerate(Y_total_list), total=len(Y_total_list)):

    # check if there is pre-trained file existed
    model_file_name = f'{model_name}_model_len_{filtered_num}_num_{random_pick_num}_{target_name}.pkl'
    model_save_file_name = os.path.join(model_save_folder, model_file_name)
    if os.path.exists(model_save_file_name):
        print(
            f"\n-----> WARNING: SKIPPING TRAINING for Target '{target_name}'. Model already saved at: {model_save_file_name}")
        continue

    y_data = Y_total_file[str(target_name)].to_numpy().ravel().astype('float32')
    # split
    X_train, X_temp, y_train, y_temp = train_test_split(X_file, y_data, test_size=0.2, random_state=777)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=777)

    # Print shapes to verify splits
    print(
        f"target: {target_name} | X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # Optuna for hyperparameters tuning
    my_model = optuna_tuning.tune_with_optuna(model_class=initial_model,
                                              model_name=model_name,
                                              X_train=X_train,
                                              y_train=y_train,
                                              X_val=X_val,
                                              y_val=y_val,
                                              n_trials=optuna_num_trials,
                                              device=device)

    print(f'Model type created: {type(my_model)}')

    # model training
    my_model.fit(X_train, y_train)
    y_pred = my_model.predict(X_test)

    # Calculate and print performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(
        f"\n----- Metrics for {model_name} Model | Target: {target_name}, len: {filtered_num}, num: {random_pick_num} -----")
    print(f"Model: {model_name} | target: {target_name} | test MSE: {mse:.3f}")
    print(f"Model: {model_name} | target: {target_name} | test MAE: {mae:.3f}")
    print(f"Model: {model_name} | target: {target_name} | test RMSE: {rmse:.3f}")
    print(f"Model: {model_name} | target: {target_name} | test R^2 Score: {r2:.3f}")
    print(f"Model: {model_name} | target: {target_name} | test MAPE: {mape:.3f}%")

    # MAE Plot show and save
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')  # Actual: y_test, Predicted: y_pred

    # Add a line of perfect prediction (y=x) for reference
    all_values = np.concatenate([y_test, y_pred])
    min_val = all_values.min() * 0.95  # margins 95%
    max_val = all_values.max() * 1.05  # margins 105%
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Perfect Prediction')

    # Add labels, title, and a legend
    plt.xlabel(f'Actual {target_name}', fontsize=12)
    plt.ylabel(f'Predicted {target_name}', fontsize=12)
    plt.title(f'{model_name}: Actual vs. Predicted ({target_name}) \n(MAE: {mae:.4f}, R2: {r2:.4f})', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plot_save_path = os.path.join(plot_save_folder, f'{model_name}_MAE_plot_{target_name}.png')
    plt.savefig(plot_save_path, bbox_inches='tight')
    plt.close()
    print(f'{model_name} MAE plot saved to {plot_save_path}')

    # Create a dictionary to hold ALL metadata and the model object
    model_export_package = {'model': model_name,
                            'target_variable': target_name,
                            'model_object': my_model,
                            'hyperparameters_used': my_model.get_params(),
                            'training_metrics': {'MSE_score': mse,
                                                 'MAE_score': mae,
                                                 'RMSE_score': rmse,
                                                 'R2_score': r2,
                                                 'MAPE_score': mape}}

    # Save model
    joblib.dump(model_export_package, model_save_file_name)
    print(f"{model_name} model and all parameters are saved to {model_save_file_name}")

print("All the training is complete!")

# To load this comprehensive package later:
# loaded_package = joblib.load('model_save_file_name.pkl')
# loaded_model = loaded_package['model_object']
# print(loaded_package['hyperparameters_used'])

