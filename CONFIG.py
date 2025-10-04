##### CONFIG.py
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
filtered_num = 30 # filtering num of SMILES
random_pick_num = 100000 # num_pick
data_extraction_folder = fr"C:\Users\wisdo\polyOne_Data_Set\data_extraction"
chemical_feature_extraction_folder = fr"C:\Users\wisdo\polyOne_Data_Set\chemical_feature_extraction"
model_save_folder = fr"C:\Users\wisdo\polyOne_Data_Set\models"
plot_save_folder = fr"C:\Users\wisdo\polyOne_Data_Set\plot"

ECFP_radius = 2
ECFP_nBits = 1024
ECFP_num = 1024

optuna_num_trials = 10
