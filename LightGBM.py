import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
import numpy as np

# Load training data and supplementary 'greeks' data.
df = pd.read_csv('train.csv', index_col=0)
greeks = pd.read_csv('greeks.csv', index_col=0)

# Ensure column names do not have leading/trailing spaces.
df.columns = df.columns.str.strip()

# Transform 'EJ' column from categorical to numeric representation.
le = LabelEncoder()
df['EJ'] = le.fit_transform(df['EJ'])

# Separate features and target variable from the dataset.
y = df['Class']
X = df.drop('Class', axis=1)

# Hyperparameters obtained from prior optimization.
params = {
    'bagging_fraction': 0.6253866957581725, 
    'bagging_freq': 4, 
    'feature_fraction': 0.5750404860707732, 
    'lambda_l1': 0.6860310523391222, 
    'lambda_l2': 13.644318182721516, 
    'learning_rate': 0.2947980311444581, 
    'max_bin': 304, 
    'max_depth': 4, 
    'min_data_in_leaf': 14, 
    'min_gain_to_split': 1.2896729508484732, 
    'min_sum_hessian_in_leaf': 0.07363553821982624, 
    'n_estimators': 511, 
    'num_leaves': 62, 
    'scale_pos_weight': 46.62739132700027
}

# Initialize the LightGBM classifier with specified hyperparameters.
model = LGBMClassifier(**params, verbose=-1, objective='binary')

# Define custom loss function to handle imbalanced classes.
def balanced_log_loss(y_true, y_pred):
    n0 = len(y_true[y_true == 0])
    n1 = len(y_true[y_true == 1])

    # Ensure we do not have any classes without samples.
    if n0 == 0:
        raise ValueError("n0 empty")
    elif n1 == 0:
        raise ValueError("n1 empty")

    # Calculate log loss for each class.
    log_loss_0 = -np.sum([np.log(1-p + 1e-15) for y, p in zip(y_true, y_pred) if y == 0]) / n0 if n0 > 0 else 1000
    log_loss_1 = -np.sum([np.log(p + 1e-15) for y, p in zip(y_true, y_pred) if y == 1]) / n1 if n1 > 0 else 1000

    return (log_loss_0 + log_loss_1) / 2

# Wrap the custom loss function in a scorer object for model evaluation.
scorer = make_scorer(balanced_log_loss, greater_is_better=False, needs_proba=True)

# Perform cross-validation with stratified sampling for evaluation.
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv_score = cross_val_score(model, X, y, cv=cv, scoring=scorer)

# Display mean of cross-validation scores.
print('CV Score:', np.mean(cv_score))
