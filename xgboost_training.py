import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score
import numpy as np
import os
import pickle
from rdkit.ML.Scoring import Scoring
import json
import shutil
import argparse


# Function to clear the contents of a directory
def clear_folder(folder_path):
    """Removes all files and directories inside a given folder."""
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


def load_data(train_features_path, train_labels_path, test_features_path, test_labels_path):
    """Loads training and test data from specified file paths."""
    X_train = pickle.load(open(train_features_path, 'rb'))
    y_train = pickle.load(open(train_labels_path, 'rb'))
    X_test = pickle.load(open(test_features_path, 'rb'))
    y_test = pickle.load(open(test_labels_path, 'rb'))

    return X_train, y_train, X_test, y_test


def main(args):
    # Load training and validation data
    print('Loading data ...')
    X_train, y_train, X_test, y_test = load_data(
        args.train_features, args.train_labels, args.test_features, args.test_labels)

    print("Train labels distribution:\n", y_train.value_counts())
    print("Test labels distribution:\n", y_test.value_counts())
    print("Train data shape: ", X_train.shape)
    print("Test data shape: ", X_test.shape)
    print('Data loaded!')

    # Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define the objective function for optimization
    def objective(trial):
        """Objective function for hyperparameter tuning."""
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
        ef = Scoring.CalcEnrichment(sorted_all_preds_trail, 0, enrichment_threshold)
        ef_dict = dict(zip(enrichment_threshold, ef))

        print(
            f"AUCPR: {aucpr}\nBEDROC: {bedroc}\nEF_0.5%: {ef_dict[0.005]:.4f}, EF_1%: {ef_dict[0.01]:.4f}, EF_2%: {ef_dict[0.02]:.4f}, EF_5%: {ef_dict[0.05]:.4f}")

        return aucpr

    # Hyperparameter optimization with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Get best parameters and update the initial ones
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    # Retrain the model using the best parameters
    final_model = xgb.train(best_params, dtrain, num_boost_round=50000, evals=[(dtrain, 'train'), (dtest, 'val')],
                            early_stopping_rounds=200, verbose_eval=True)

    # Evaluate the final model on the test set
    preds = final_model.predict(dtest)
    aucpr = average_precision_score(y_test, preds)

    all_preds_np = np.array(preds).reshape(-1, 1)
    all_labels_np = np.array(y_test).reshape(-1, 1)

    # Concatenate predictions and labels
    all_preds_trail = np.concatenate((all_labels_np, all_preds_np), axis=1)
    sorted_indices = np.argsort(-all_preds_np, axis=0)  # Sort in descending order
    sorted_all_preds_trail = all_preds_trail[sorted_indices[:, 0]]

    # Calculate performance metrics
    bedroc = Scoring.CalcBEDROC(sorted_all_preds_trail, 0, alpha=80.5)
    ef_1_percent = Scoring.CalcEnrichment(sorted_all_preds_trail, 0, [0.01])
    ef_5_percent = Scoring.CalcEnrichment(sorted_all_preds_trail, 0, [0.05])

    print(f"AUCPR: {aucpr}\nBEDROC: {bedroc}\nEF_1%: {ef_1_percent[0]:.4f}, EF_5%: {ef_5_percent[0]:.4f}")

    # Save the final model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    final_model_path = os.path.join(args.output_dir, f'SC2_weight_{bedroc:.3f}_aucpr_{aucpr:.3f}.pkl')
    pickle.dump(final_model, open(final_model_path, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an XGBoost model with hyperparameter optimization.')
    parser.add_argument('--train_features', type=str, required=True, help='Path to the training features file')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to the training labels file')
    parser.add_argument('--test_features', type=str, required=True, help='Path to the test features file')
    parser.add_argument('--test_labels', type=str, required=True, help='Path to the test labels file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the trained model')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for hyperparameter optimization')

    args = parser.parse_args()
    main(args)
