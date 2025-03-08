import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score
import numpy as np
import os
import pickle
from rdkit.ML.Scoring import Scoring
import json
import shutil

# Clear folder function to clean up model saving directory
def clear_folder(folder_path):
    try:
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                os.makedirs(item_path, exist_ok=True)
        print(f"Cleared folder: {folder_path}")
    except Exception as e:
        os.makedirs(folder_path, exist_ok=True)


# Load training and validation data
print('Loading data ...')

X_train = pickle.load(open('/home/s2523227/pdbscreen/features_deduplicated_combine_testset_valset_as_testset/train_features_normalized.pkl','rb'))
y_train = pickle.load(open('/home/s2523227/pdbscreen/features_deduplicated_combine_testset_valset_as_testset/pdbscreening_train_labels.pkl', 'rb'))
X_test = pickle.load(open('/home/s2523227/pdbscreen/features_deduplicated_combine_testset_valset_as_testset/test_features_normalized.pkl','rb'))
y_test = pickle.load(open('/home/s2523227/pdbscreen/features_deduplicated_combine_testset_valset_as_testset/pdbscreening_test_labels.pkl', 'rb'))

output_dir = '/home/s2523227/sc2_dapanther/weight/classification_weight'


print(y_train.value_counts())
print(y_test.value_counts())
print("train_df shape: ", X_train.shape)
print("test_df shape: ", X_test.shape)
print('Data loaded!')

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)



# Define the objective function for optimization
def objective(trial):

    param = {
        'tree_method': 'hist',
        'device': 'cuda',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0, step=0.01),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 30),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'n_estimators': trial.suggest_int('n_estimators', 1, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1),
        'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train),
        'objective': 'binary:logistic',
    }

    enrichment_threshold = [0.005, 0.01, 0.02, 0.05]
    evals = [(dtrain, 'train'), (dtest, 'test')]

    # Train the model with updated parameters
    model = xgb.train(param, dtrain,
                      num_boost_round=2000,
                      evals=evals,
                      early_stopping_rounds=50,
                      verbose_eval=True)

    # Make predictions and calculate performance metrics
    preds = model.predict(dtest)
    all_preds_np = np.array(preds).reshape(-1, 1)
    all_labels_np = np.array(y_test).reshape(-1, 1)

    # Concatenate predictions and labels
    all_preds_trail = np.concatenate((all_labels_np, all_preds_np), axis=1)
    sorted_indices = np.argsort(-all_preds_np, axis=0)  # Sort in descending order
    sorted_all_preds_trail = all_preds_trail[sorted_indices[:, 0]]

    # Calculate AUCPR, BEDROC, and EF 1%
    aucpr = average_precision_score(y_test, preds)
    bedroc = Scoring.CalcBEDROC(sorted_all_preds_trail, 0, alpha=80.5)
    # ef = Scoring.CalcEnrichment(sorted_all_preds_trail, 0, [0.01,0.05])
    ef = Scoring.CalcEnrichment(sorted_all_preds_trail, 0, enrichment_threshold)
    ef_dict = dict(zip(enrichment_threshold, ef))


    print(f"AUCPR: {aucpr}\nBEDROC: {bedroc}\nEF_0.5%: {ef_dict[0.005]:.4f}, EF_1%: {ef_dict[0.01]:.4f}, EF_2%: {ef_dict[0.02]:.4f}, EF_5%: {ef_dict[0.05]:.4f}")

    return aucpr


# Hyperparameter optimization with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)  # Increase n_trials for better optimization

# Get best parameters and update the initial ones
best_params = study.best_params
best_params['tree_method'] = 'gpu_hist'
best_params['scale_pos_weight'] = (len(y_train) - sum(y_train)) / sum(y_train)
best_params['objective'] = 'binary:logistic'
print(f"Best hyperparameters: {best_params}")

# Retrain the model using the best parameters
final_model = xgb.train(best_params, dtrain, num_boost_round=50000, evals=[(dtrain, 'train'), (dtest, 'val')],
                        early_stopping_rounds=200, verbose_eval=True)

# Evaluate the final model on the validation set
preds = final_model.predict(dtest)
aucpr = average_precision_score(y_test, preds)

all_preds_np = np.array(preds).reshape(-1, 1)
all_labels_np = np.array(y_test).reshape(-1, 1)

# Concatenate predictions and labels
all_preds_trail = np.concatenate((all_labels_np, all_preds_np), axis=1)
sorted_indices = np.argsort(-all_preds_np, axis=0)  # Sort in descending order
sorted_all_preds_trail = all_preds_trail[sorted_indices[:, 0]]

bedroc = Scoring.CalcBEDROC(sorted_all_preds_trail, 0, alpha=80.5)
ef_1_percent = Scoring.CalcEnrichment(sorted_all_preds_trail, 0, [0.01])
ef_5_percent = Scoring.CalcEnrichment(sorted_all_preds_trail, 0, [0.05])

print(f"AUCPR: {aucpr}\nBEDROC: {bedroc}\nEF_1%: {ef_1_percent[0]:.4f}, EF_5%: {ef_5_percent[0]:.4f}")

# Save the final model
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

final_model_path = os.path.join(output_dir,f'SC2_PS_deduplicated_combine_testset_valset_as_testset_bedroc_{bedroc:.3f}_aucpr_{aucpr:.3f}.pkl')
pickle.dump(final_model, open(final_model_path, 'wb'))






