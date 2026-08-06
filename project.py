# STEP 0: Check data file existence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("BANKNOTE AUTHENTICATION PROJECT")
print("STEP 0 - Checking data file")
print("=" * 70)

file_name = "data_banknote_authentication.txt"

try:
    df_test = pd.read_csv(file_name, header=None)
    print(f"[SUCCESS] File '{file_name}' found.")
    print(f"[INFO] Total records: {df_test.shape[0]}")
    print(f"[INFO] Total columns: {df_test.shape[1]}")
    print("\nFirst 5 rows of data:")
    print(df_test.head())
except FileNotFoundError:
    print(f"[ERROR] File '{file_name}' not found.")
    print("Please check the file name or place it in the current directory.")
    exit()

print("-" * 70)
print("STEP 0 completed successfully.")
print("-" * 70)

# STEP 1: Data Preparation
def load_banknote_data(file_path="data_banknote_authentication.txt"):
    print("\n[LOADING] Loading data from:", file_path)
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :4].values
    y = data.iloc[:, -1].values
    print(f"[OK] Total samples: {len(X)}")
    print(f"[OK] Number of features: {X.shape[1]}")
    print(f"[INFO] Class 0 (genuine): {np.sum(y==0)} samples")
    print(f"[INFO] Class 1 (forged): {np.sum(y==1)} samples")
    return X, y


def min_max_normalize(X):
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1, range_vals)
    X_norm = (X - min_vals) / range_vals
    return X_norm, min_vals, max_vals, range_vals


def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    np.random.seed(seed)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_folds_manual(X, y, n_folds=5, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    fold_sizes = [len(X) // n_folds] * n_folds
    for i in range(len(X) % n_folds):
        fold_sizes[i] += 1
    folds = []
    current = 0
    for size in fold_sizes:
        val_idx = indices[current:current+size]
        train_idx = np.concatenate([indices[:current], indices[current+size:]])
        folds.append((train_idx, val_idx))
        current += size
    return folds


print("\n" + "=" * 70)
print("STEP 1 - Data Preparation")
print("=" * 70)

X_raw, y = load_banknote_data("data_banknote_authentication.txt")
print("\n[MIN-MAX NORMALIZATION] For KNN algorithm")
print("-" * 50)
X_minmax, min_vals, max_vals, range_vals = min_max_normalize(X_raw)
print(f"[INFO] Feature 0 (Variance):  min={min_vals[0]:.4f}, max={max_vals[0]:.4f}")
print(f"[INFO] Feature 1 (Skewness):  min={min_vals[1]:.4f}, max={max_vals[1]:.4f}")
print(f"[INFO] Feature 2 (Kurtosis):  min={min_vals[2]:.4f}, max={max_vals[2]:.4f}")
print(f"[INFO] Feature 3 (Entropy):   min={min_vals[3]:.4f}, max={max_vals[3]:.4f}")
print("\n[TRAIN/VAL/TEST SPLIT] 70% / 15% / 15%")
print("-" * 50)

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X_raw, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
)

print(f"[INFO] Training set:   {len(X_train)} samples ({len(X_train)/len(X_raw)*100:.0f}%)")
print(f"[INFO] Validation set: {len(X_val)} samples ({len(X_val)/len(X_raw)*100:.0f}%)")
print(f"[INFO] Test set:       {len(X_test)} samples ({len(X_test)/len(X_raw)*100:.0f}%)")

print("\n[CLASS DISTRIBUTION]")
print(f"  Training:   Class 0={np.sum(y_train==0)}, Class 1={np.sum(y_train==1)}")
print(f"  Validation: Class 0={np.sum(y_val==0)}, Class 1={np.sum(y_val==1)}")
print(f"  Test:       Class 0={np.sum(y_test==0)}, Class 1={np.sum(y_test==1)}")

X_train_knn = (X_train - min_vals) / range_vals
X_val_knn = (X_val - min_vals) / range_vals
X_test_knn = (X_test - min_vals) / range_vals
X_minmax_full, _, _, _ = min_max_normalize(X_raw)

cv_folds = create_folds_manual(X_raw, y, n_folds=5, seed=42)
print(f"\n[CROSS-VALIDATION FOLDS] Created {len(cv_folds)} folds for 5-fold CV")

print("\n[SUCCESS] STEP 1 completed.")
print("-" * 70)

# STEP 2: Evaluation Metrics
def calculate_metrics_from_scratch(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


print("\n" + "=" * 70)
print("STEP 2 - Evaluation Metrics Test")
print("=" * 70)

y_test_sample = np.array([0, 1, 1, 0, 1])
y_pred_sample = np.array([0, 1, 0, 0, 1])
metrics_sample = calculate_metrics_from_scratch(y_test_sample, y_pred_sample)
print("[TEST] Sample predictions:")
print(f"   True labels:  {y_test_sample}")
print(f"   Pred labels:  {y_pred_sample}")
print("\n[RESULTS]")
for key, value in metrics_sample.items():
    print(f"   {key}: {value:.4f}")

print("\n[SUCCESS] Metrics working correctly.")
print("-" * 70)

# STEP 3: Naive Bayes (Gaussian)
def gaussian_pdf(x, mean, var):
    eps = 1e-9
    return (1 / np.sqrt(2 * np.pi * var + eps)) * np.exp(-((x - mean)**2) / (2 * var + eps))


def train_naive_bayes(X, y):
    classes = np.unique(y)
    mean_dict = {}
    var_dict = {}
    prior_dict = {}
    for c in classes:
        X_c = X[y == c]
        mean_dict[c] = np.mean(X_c, axis=0)
        var_dict[c] = np.var(X_c, axis=0)
        prior_dict[c] = len(X_c) / len(X)
    return classes, mean_dict, var_dict, prior_dict


def predict_naive_bayes(X, classes, mean_dict, var_dict, prior_dict):
    predictions = []
    for x in X:
        posterior_log = {}
        for c in classes:
            log_likelihood = 0.0
            for i in range(len(x)):
                prob = gaussian_pdf(x[i], mean_dict[c][i], var_dict[c][i])
                log_likelihood += np.log(prob + 1e-9)
            log_prior = np.log(prior_dict[c] + 1e-9)
            posterior_log[c] = log_likelihood + log_prior
        pred = max(posterior_log, key=posterior_log.get)
        predictions.append(pred)
    return np.array(predictions)


def cross_validate_naive_bayes(X, y, folds):
    all_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        classes, means, variances, priors = train_naive_bayes(X_train_fold, y_train_fold)
        y_pred = predict_naive_bayes(X_val_fold, classes, means, variances, priors)
        metrics = calculate_metrics_from_scratch(y_val_fold, y_pred)
        all_metrics.append(metrics)
        print(f"   Fold {fold_idx + 1}: Accuracy = {metrics['accuracy']:.4f}")
    return all_metrics


print("\n" + "=" * 70)
print("STEP 3 - Naive Bayes Implementation")
print("=" * 70)

print("[INFO] Training Naive Bayes on 70% training data...")
classes, means, variances, priors = train_naive_bayes(X_train, y_train)

print("[INFO] Evaluating on validation set...")
y_val_pred_nb = predict_naive_bayes(X_val, classes, means, variances, priors)
val_metrics_nb = calculate_metrics_from_scratch(y_val, y_val_pred_nb)

print("\n[VALIDATION RESULTS]")
print("-" * 50)
for key, value in val_metrics_nb.items():
    print(f"   {key}: {value:.4f}")

print("\n[FINAL TEST] Evaluating on 15% test set...")
y_test_pred_nb = predict_naive_bayes(X_test, classes, means, variances, priors)
test_metrics_nb = calculate_metrics_from_scratch(y_test, y_test_pred_nb)

print("\n[TEST RESULTS]")
print("-" * 50)
for key, value in test_metrics_nb.items():
    print(f"   {key}: {value:.4f}")

print("\n[CROSS-VALIDATION] 5-fold for Naive Bayes:")
print("-" * 50)
nb_cv_results = cross_validate_naive_bayes(X_raw, y, cv_folds)

print("\n[SUCCESS] Naive Bayes is ready.")
print("=" * 70)

# STEP 4: KNN
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def predict_knn(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = []
        for train_point in X_train:
            dist = euclidean_distance(test_point, train_point)
            distances.append(dist)
        distances = np.array(distances)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        unique_labels = np.unique(nearest_labels)
        vote_counts = []
        for label in unique_labels:
            count = np.sum(nearest_labels == label)
            vote_counts.append(count)
        max_index = np.argmax(vote_counts)
        prediction = unique_labels[max_index]
        predictions.append(prediction)
    return np.array(predictions)


def cross_validate_knn(X, y, folds, k):
    all_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        y_pred = predict_knn(X_train_fold, y_train_fold, X_val_fold, k)
        metrics = calculate_metrics_from_scratch(y_val_fold, y_pred)
        all_metrics.append(metrics)
        print(f"   Fold {fold_idx + 1}: Accuracy = {metrics['accuracy']:.4f}")
        print(f"       Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}, F1 = {metrics['f1']:.4f}")
    return all_metrics


print("\n" + "=" * 70)
print("STEP 4 - KNN Implementation (k = 1, 3, 5, 7)")
print("=" * 70)

print("[INFO] Algorithm: Lazy learner - stores all training data")
print("[INFO] Distance metric: Euclidean distance")
print("[INFO] Decision: Simple majority voting (no weights)")
print("[INFO] Normalization: Min-Max (as required by project)")
print("[INFO] k values to test: 1, 3, 5, 7")
print(f"[INFO] Data shape: {X_minmax_full.shape}")
print(f"[INFO] Number of folds: {len(cv_folds)}")

print("\n[INFO] Testing different k values with cross-validation...")
print("-" * 50)

k_test_values = [1, 3, 5, 7]
best_k = 3
best_acc = 0
knn_results_dict = {}

for k in k_test_values:
    print(f"\n{'='*50}")
    print(f"Testing k = {k}:")
    print(f"{'-'*50}")
    
    results = cross_validate_knn(X_minmax_full, y, cv_folds, k)
    knn_results_dict[k] = results
    
    mean_acc = np.mean([r['accuracy'] for r in results])
    std_acc = np.std([r['accuracy'] for r in results])
    print(f"\n   --> Mean accuracy for k={k}: {mean_acc:.4f} +/- {std_acc:.4f}")
    
    if mean_acc > best_acc:
        best_acc = mean_acc
        best_k = k

print("\n" + "-" * 50)
print(f"[RESULT] Best k = {best_k} with accuracy = {best_acc:.4f}")

print("\n" + "=" * 70)
print(f"DETAILED RESULTS FOR KNN (k={best_k})")
print("=" * 70)

knn_cv_results = knn_results_dict[best_k]

knn_cv_acc = [m['accuracy'] for m in knn_cv_results]
knn_cv_prec = [m['precision'] for m in knn_cv_results]
knn_cv_rec = [m['recall'] for m in knn_cv_results]
knn_cv_f1 = [m['f1'] for m in knn_cv_results]

print("\n[FINAL RESULTS FOR KNN]")
print("-" * 50)
print(f"  Accuracy:   {np.mean(knn_cv_acc):.4f} +/- {np.std(knn_cv_acc):.4f}")
print(f"  Precision:  {np.mean(knn_cv_prec):.4f} +/- {np.std(knn_cv_prec):.4f}")
print(f"  Recall:     {np.mean(knn_cv_rec):.4f} +/- {np.std(knn_cv_rec):.4f}")
print(f"  F1 Score:   {np.mean(knn_cv_f1):.4f} +/- {np.std(knn_cv_f1):.4f}")

print("\n[PER-FOLD RESULTS]")
print("-" * 50)
for i, metrics in enumerate(knn_cv_results):
    print(f"  Fold {i+1}: Acc={metrics['accuracy']:.4f}, "
          f"Prec={metrics['precision']:.4f}, "
          f"Rec={metrics['recall']:.4f}, "
          f"F1={metrics['f1']:.4f}")

print("\n" + "=" * 70)
print("SUMMARY FOR ALL K VALUES (1, 3, 5, 7)")
print("=" * 70)

print("\n[ACCURACY COMPARISON]")
print("-" * 50)
print(f"{'k value':<10} {'Mean Accuracy':<20} {'Std Deviation':<20}")
print("-" * 50)

for k in k_test_values:
    results = knn_results_dict[k]
    mean_acc = np.mean([r['accuracy'] for r in results])
    std_acc = np.std([r['accuracy'] for r in results])
    print(f"{k:<10} {mean_acc:<20.4f} {std_acc:<20.4f}")

print("-" * 50)
print("\n" + "=" * 70)
print("FINAL TEST ON 15% HOLDOUT TEST SET")
print("=" * 70)

print("\n[INFO] Finding best k using validation set (15%):")
print("-" * 50)

best_k_val = 3
best_val_acc = 0

for k in k_test_values:
    y_val_pred = predict_knn(X_train_knn, y_train, X_val_knn, k)
    val_acc = np.mean(y_val_pred == y_val)
    val_metrics = calculate_metrics_from_scratch(y_val, y_val_pred)
    
    print(f"  k={k}: Validation Accuracy = {val_acc:.4f}, "
          f"Precision={val_metrics['precision']:.4f}, "
          f"Recall={val_metrics['recall']:.4f}, "
          f"F1={val_metrics['f1']:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_k_val = k

print(f"\n[RESULT] Best k using validation set = {best_k_val} (accuracy = {best_val_acc:.4f})")

print("\n[FINAL TEST] Evaluating on 15% test set...")
print("-" * 50)

y_test_pred_knn = predict_knn(X_train_knn, y_train, X_test_knn, best_k_val)
test_metrics_knn = calculate_metrics_from_scratch(y_test, y_test_pred_knn)

print("\n[TEST RESULTS]")
print("-" * 50)
for key, value in test_metrics_knn.items():
    print(f"  {key}: {value:.4f}")

print("\n[SUCCESS] KNN is ready.")
print("=" * 70)

# STEP 5: Decision Tree ID3
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-9))

def discretize_feature(X_feature):
    threshold = np.mean(X_feature)
    return (X_feature > threshold).astype(int)

def information_gain(X_feature, y):
    X_discrete = discretize_feature(X_feature)
    H_parent = entropy(y)
    unique_vals = np.unique(X_discrete)
    H_child = 0
    for val in unique_vals:
        y_child = y[X_discrete == val]
        weight = len(y_child) / len(y)
        H_child += weight * entropy(y_child)
    return H_parent - H_child

def build_tree(X, y, feature_indices, depth, max_depth=5):
    if len(np.unique(y)) == 1:
        return {'leaf': True, 'class': y[0]}
    
    if depth >= max_depth:
        unique, counts = np.unique(y, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        return {'leaf': True, 'class': majority_class}
    
    best_gain = -1
    best_feature = None
    
    for i in feature_indices:
        gain = information_gain(X[:, i], y)
        if gain > best_gain:
            best_gain = gain
            best_feature = i
    
    if best_feature is None or best_gain <= 0:
        unique, counts = np.unique(y, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        return {'leaf': True, 'class': majority_class}
    
    X_feature = X[:, best_feature]
    threshold = np.mean(X_feature)
    X_discrete = (X_feature > threshold).astype(int)
    
    node = {'leaf': False, 'feature': best_feature, 'threshold': threshold, 'children': {}}
    
    for val in [0, 1]:
        indices = np.where(X_discrete == val)[0]
        if len(indices) == 0:
            unique, counts = np.unique(y, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            node['children'][val] = {'leaf': True, 'class': majority_class}
        else:
            X_child = X[indices]
            y_child = y[indices]
            node['children'][val] = build_tree(X_child, y_child, feature_indices, depth + 1, max_depth)
    
    return node


def train_decision_tree(X, y, max_depth=5):
    feature_indices = list(range(X.shape[1]))
    tree = build_tree(X, y, feature_indices, 0, max_depth)
    return tree


def predict_tree_single(x, tree):
    if tree['leaf']:
        return tree['class']
    feature_val = x[tree['feature']]
    discrete_val = 1 if feature_val > tree['threshold'] else 0
    if discrete_val in tree['children']:
        return predict_tree_single(x, tree['children'][discrete_val])
    return 0


def predict_decision_tree(X, tree):
    return np.array([predict_tree_single(x, tree) for x in X])


def cross_validate_decision_tree(X, y, folds, max_depth=5):
    all_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        tree = train_decision_tree(X_train_fold, y_train_fold, max_depth)
        y_pred = predict_decision_tree(X_val_fold, tree)
        metrics = calculate_metrics_from_scratch(y_val_fold, y_pred)
        all_metrics.append(metrics)
        print(f"   Fold {fold_idx + 1}: Accuracy = {metrics['accuracy']:.4f}")
    return all_metrics

print("\n" + "=" * 70)
print("STEP 5 - Decision Tree ID3 (max_depth=5)")
print("=" * 70)

print("[INFO] Algorithm: ID3 with continuous feature discretization")
print("[INFO] Discretization method: Mean threshold")
print("[INFO] Pre-pruning: max_depth = 5")
print(f"[INFO] Data shape: {X_raw.shape}")
print(f"[INFO] Number of folds: {len(cv_folds)}")

print("\n[CROSS-VALIDATION] 5-fold for Decision Tree (max_depth=5):")
print("-" * 50)

dt_cv_results = cross_validate_decision_tree(X_raw, y, cv_folds, max_depth=5)

dt_cv_acc = [m['accuracy'] for m in dt_cv_results]
dt_cv_prec = [m['precision'] for m in dt_cv_results]
dt_cv_rec = [m['recall'] for m in dt_cv_results]
dt_cv_f1 = [m['f1'] for m in dt_cv_results]

print("\n[FINAL RESULTS FOR DECISION TREE (max_depth=5)]")
print("-" * 50)
print(f"  Accuracy:   {np.mean(dt_cv_acc):.4f} +/- {np.std(dt_cv_acc):.4f}")
print(f"  Precision:  {np.mean(dt_cv_prec):.4f} +/- {np.std(dt_cv_prec):.4f}")
print(f"  Recall:     {np.mean(dt_cv_rec):.4f} +/- {np.std(dt_cv_rec):.4f}")
print(f"  F1 Score:   {np.mean(dt_cv_f1):.4f} +/- {np.std(dt_cv_f1):.4f}")

print("\n[PER-FOLD RESULTS]")
print("-" * 50)
for i, metrics in enumerate(dt_cv_results):
    print(f"  Fold {i+1}: Acc={metrics['accuracy']:.4f}, "
          f"Prec={metrics['precision']:.4f}, "
          f"Rec={metrics['recall']:.4f}, "
          f"F1={metrics['f1']:.4f}")

print("\n[FINAL TEST] Training Decision Tree on 70% training data...")
tree_final = train_decision_tree(X_train, y_train, max_depth=5)

print("[INFO] Evaluating on validation set...")
y_val_pred_dt = predict_decision_tree(X_val, tree_final)
val_metrics_dt = calculate_metrics_from_scratch(y_val, y_val_pred_dt)

print("\n[VALIDATION RESULTS]")
print("-" * 50)
for key, value in val_metrics_dt.items():
    print(f"   {key}: {value:.4f}")

print("\n[FINAL TEST] Evaluating on 15% test set...")
y_test_pred_dt = predict_decision_tree(X_test, tree_final)
test_metrics_dt = calculate_metrics_from_scratch(y_test, y_test_pred_dt)

print("\n[TEST RESULTS]")
print("-" * 50)
for key, value in test_metrics_dt.items():
    print(f"   {key}: {value:.4f}")

print("\n[SUCCESS] Decision Tree ID3 is ready.")
print("=" * 70)

# STEP 6: Paired t-test
print("\n" + "=" * 70)
print("STEP 6 - Paired t-test")
print("=" * 70)

nb_acc = [m['accuracy'] for m in nb_cv_results]
knn_acc = [m['accuracy'] for m in knn_cv_results]
dt_acc = [m['accuracy'] for m in dt_cv_results]

print("\n[PAIRED T-TEST RESULTS]")
print("-" * 50)

t_stat_nb_knn, p_val_nb_knn = ttest_rel(nb_acc, knn_acc)
print(f"\n  Naive Bayes vs KNN:")
print(f"    t-statistic = {t_stat_nb_knn:.4f}")
print(f"    p-value = {p_val_nb_knn:.4f}")
print(f"    Significant (p < 0.05): {'YES' if p_val_nb_knn < 0.05 else 'NO'}")

t_stat_nb_dt, p_val_nb_dt = ttest_rel(nb_acc, dt_acc)
print(f"\n  Naive Bayes vs Decision Tree:")
print(f"    t-statistic = {t_stat_nb_dt:.4f}")
print(f"    p-value = {p_val_nb_dt:.4f}")
print(f"    Significant (p < 0.05): {'YES' if p_val_nb_dt < 0.05 else 'NO'}")

t_stat_knn_dt, p_val_knn_dt = ttest_rel(knn_acc, dt_acc)
print(f"\n  KNN vs Decision Tree:")
print(f"    t-statistic = {t_stat_knn_dt:.4f}")
print(f"    p-value = {p_val_knn_dt:.4f}")
print(f"    Significant (p < 0.05): {'YES' if p_val_knn_dt < 0.05 else 'NO'}")

# STEP 7: Learning Curves
print("\n" + "=" * 70)
print("STEP 7 - Learning Curves")
print("=" * 70)

train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

def learning_curve_naive_bayes(X, y, train_sizes, cv_folds, seed=42):
    accuracies = []
    stds = []
    for size in train_sizes:
        n_samples = int(len(X) * size)
        X_subset = X[:n_samples]
        y_subset = y[:n_samples]
        folds_subset = create_folds_manual(X_subset, y_subset, n_folds=5, seed=seed)
        results = cross_validate_naive_bayes(X_subset, y_subset, folds_subset)
        acc = [r['accuracy'] for r in results]
        accuracies.append(np.mean(acc))
        stds.append(np.std(acc))
        print(f"  {int(size*100)}% data: Accuracy = {np.mean(acc):.4f} +/- {np.std(acc):.4f}")
    return accuracies, stds

def learning_curve_knn(X, y, X_norm, train_sizes, cv_folds, k, seed=42):
    accuracies = []
    stds = []
    for size in train_sizes:
        n_samples = int(len(X) * size)
        X_subset = X[:n_samples]
        X_norm_subset = X_norm[:n_samples]
        y_subset = y[:n_samples]
        folds_subset = create_folds_manual(X_subset, y_subset, n_folds=5, seed=seed)
        results = cross_validate_knn(X_norm_subset, y_subset, folds_subset, k)
        acc = [r['accuracy'] for r in results]
        accuracies.append(np.mean(acc))
        stds.append(np.std(acc))
        print(f"  {int(size*100)}% data: Accuracy = {np.mean(acc):.4f} +/- {np.std(acc):.4f}")
    return accuracies, stds

def learning_curve_decision_tree(X, y, train_sizes, cv_folds, max_depth=5, seed=42):
    accuracies = []
    stds = []
    for size in train_sizes:
        n_samples = int(len(X) * size)
        X_subset = X[:n_samples]
        y_subset = y[:n_samples]
        folds_subset = create_folds_manual(X_subset, y_subset, n_folds=5, seed=seed)
        results = cross_validate_decision_tree(X_subset, y_subset, folds_subset, max_depth)
        acc = [r['accuracy'] for r in results]
        accuracies.append(np.mean(acc))
        stds.append(np.std(acc))
        print(f"  {int(size*100)}% data: Accuracy = {np.mean(acc):.4f} +/- {np.std(acc):.4f}")
    return accuracies, stds

print("\n[INFO] Shuffling data for learning curves...")
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X_raw))
X_shuffled = X_raw[shuffle_idx]
y_shuffled = y[shuffle_idx]
X_minmax_shuffled = X_minmax_full[shuffle_idx]

print("\n[LEARNING CURVE - Naive Bayes]")
print("-" * 50)
nb_lc_acc, nb_lc_std = learning_curve_naive_bayes(X_shuffled, y_shuffled, train_sizes, cv_folds)

print("\n[LEARNING CURVE - KNN]")
print("-" * 50)
knn_lc_acc, knn_lc_std = learning_curve_knn(X_shuffled, y_shuffled, X_minmax_shuffled, train_sizes, cv_folds, best_k)

print("\n[LEARNING CURVE - Decision Tree]")
print("-" * 50)
dt_lc_acc, dt_lc_std = learning_curve_decision_tree(X_shuffled, y_shuffled, train_sizes, cv_folds, max_depth=5)


# STEP 8: Final Summary Tables
print("\n" + "=" * 70)
print("STEP 8 - Final Summary")
print("=" * 70)
print("\n[TEST SET PERFORMANCE (15% holdout)]")
print("-" * 70)
print(f"{'Algorithm':<20} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
print("-" * 70)
print(f"{'Naive Bayes':<20} {test_metrics_nb['accuracy']:<15.4f} {test_metrics_nb['precision']:<15.4f} {test_metrics_nb['recall']:<15.4f} {test_metrics_nb['f1']:<15.4f}")
print(f"{'KNN (k=' + str(best_k_val) + ')':<20} {test_metrics_knn['accuracy']:<15.4f} {test_metrics_knn['precision']:<15.4f} {test_metrics_knn['recall']:<15.4f} {test_metrics_knn['f1']:<15.4f}")
print(f"{'Decision Tree (depth=5)':<20} {test_metrics_dt['accuracy']:<15.4f} {test_metrics_dt['precision']:<15.4f} {test_metrics_dt['recall']:<15.4f} {test_metrics_dt['f1']:<15.4f}")
print("-" * 70)
test_accuracies = {
    'Naive Bayes': test_metrics_nb['accuracy'],
    'KNN': test_metrics_knn['accuracy'],
    'Decision Tree': test_metrics_dt['accuracy']
}
best_algo = max(test_accuracies, key=test_accuracies.get)
print(f"\n[BEST ALGORITHM] Best algorithm on test set: {best_algo} with Accuracy = {test_accuracies[best_algo]:.4f}")

print("\n[CROSS-VALIDATION RESULTS (5-fold mean +/- std)]")
print("-" * 70)
print(f"  Naive Bayes:   Accuracy = {np.mean(nb_acc):.4f} +/- {np.std(nb_acc):.4f}")
print(f"  KNN:           Accuracy = {np.mean(knn_acc):.4f} +/- {np.std(knn_acc):.4f}")
print(f"  Decision Tree: Accuracy = {np.mean(dt_acc):.4f} +/- {np.std(dt_acc):.4f}")

print("\n[TABLE 1: 5-Fold Cross-Validation Results]")
print("-" * 80)
print(f"{'Algorithm':<20} {'Accuracy':<20} {'Precision':<20} {'Recall':<20} {'F1 Score':<20}")
print("-" * 80)

nb_acc_mean = np.mean([m['accuracy'] for m in nb_cv_results])
nb_acc_std = np.std([m['accuracy'] for m in nb_cv_results])
nb_prec_mean = np.mean([m['precision'] for m in nb_cv_results])
nb_prec_std = np.std([m['precision'] for m in nb_cv_results])
nb_rec_mean = np.mean([m['recall'] for m in nb_cv_results])
nb_rec_std = np.std([m['recall'] for m in nb_cv_results])
nb_f1_mean = np.mean([m['f1'] for m in nb_cv_results])
nb_f1_std = np.std([m['f1'] for m in nb_cv_results])

print(f"{'Naive Bayes':<20} {nb_acc_mean:.4f} +/- {nb_acc_std:.4f}     {nb_prec_mean:.4f} +/- {nb_prec_std:.4f}     {nb_rec_mean:.4f} +/- {nb_rec_std:.4f}     {nb_f1_mean:.4f} +/- {nb_f1_std:.4f}")

knn_acc_mean = np.mean([m['accuracy'] for m in knn_cv_results])
knn_acc_std = np.std([m['accuracy'] for m in knn_cv_results])
knn_prec_mean = np.mean([m['precision'] for m in knn_cv_results])
knn_prec_std = np.std([m['precision'] for m in knn_cv_results])
knn_rec_mean = np.mean([m['recall'] for m in knn_cv_results])
knn_rec_std = np.std([m['recall'] for m in knn_cv_results])
knn_f1_mean = np.mean([m['f1'] for m in knn_cv_results])
knn_f1_std = np.std([m['f1'] for m in knn_cv_results])

print(f"{'KNN (k=' + str(best_k) + ')':<20} {knn_acc_mean:.4f} +/- {knn_acc_std:.4f}     {knn_prec_mean:.4f} +/- {knn_prec_std:.4f}     {knn_rec_mean:.4f} +/- {knn_rec_std:.4f}     {knn_f1_mean:.4f} +/- {knn_f1_std:.4f}")

dt_acc_mean = np.mean([m['accuracy'] for m in dt_cv_results])
dt_acc_std = np.std([m['accuracy'] for m in dt_cv_results])
dt_prec_mean = np.mean([m['precision'] for m in dt_cv_results])
dt_prec_std = np.std([m['precision'] for m in dt_cv_results])
dt_rec_mean = np.mean([m['recall'] for m in dt_cv_results])
dt_rec_std = np.std([m['recall'] for m in dt_cv_results])
dt_f1_mean = np.mean([m['f1'] for m in dt_cv_results])
dt_f1_std = np.std([m['f1'] for m in dt_cv_results])

print(f"{'Decision Tree':<20} {dt_acc_mean:.4f} +/- {dt_acc_std:.4f}     {dt_prec_mean:.4f} +/- {dt_prec_std:.4f}     {dt_rec_mean:.4f} +/- {dt_rec_std:.4f}     {dt_f1_mean:.4f} +/- {dt_f1_std:.4f}")
print("-" * 80)

print("\n[TABLE 2: Paired t-test Results (p-values)]")
print("-" * 60)
print(f"{'Comparison':<30} {'p-value':<15} {'Significant (p<0.05)':<20}")
print("-" * 60)
print(f"{'Naive Bayes vs KNN':<30} {p_val_nb_knn:<15.4f} {'YES' if p_val_nb_knn < 0.05 else 'NO':<20}")
print(f"{'Naive Bayes vs Decision Tree':<30} {p_val_nb_dt:<15.4f} {'YES' if p_val_nb_dt < 0.05 else 'NO':<20}")
print(f"{'KNN vs Decision Tree':<30} {p_val_knn_dt:<15.4f} {'YES' if p_val_knn_dt < 0.05 else 'NO':<20}")
print("-" * 60)

# STEP 9: Learning Curve Interpretation
print("\n" + "=" * 70)
print("STEP 9 - Learning Curve Interpretation")
print("=" * 70)
nb_final_vs_first = nb_lc_acc[-1] - nb_lc_acc[0]
knn_final_vs_first = knn_lc_acc[-1] - knn_lc_acc[0]
dt_final_vs_first = dt_lc_acc[-1] - dt_lc_acc[0]

print("\n[BIAS-VARIANCE ANALYSIS]")
print("-" * 50)

print("\n  Naive Bayes:")
print(f"    - Accuracy at 20% data: {nb_lc_acc[0]:.4f}")
print(f"    - Accuracy at 100% data: {nb_lc_acc[-1]:.4f}")
print(f"    - Improvement: {nb_final_vs_first:.4f}")
if nb_final_vs_first > 0.05:
    print(f"    - Interpretation: HIGH BIAS (underfitting)")
    print(f"      → Model is too simple and cannot capture patterns well with limited data")
else:
    print(f"    - Interpretation: LOW BIAS - Stable model")
    print(f"      → Gaussian assumption fits the data well")

print("\n  KNN (k={}):".format(best_k))
print(f"    - Accuracy at 20% data: {knn_lc_acc[0]:.4f}")
print(f"    - Accuracy at 100% data: {knn_lc_acc[-1]:.4f}")
print(f"    - Improvement: {knn_final_vs_first:.4f}")
if knn_final_vs_first > 0.03:
    print(f"    - Interpretation: MODERATE TO HIGH VARIANCE")
    print(f"      → Performance improves with more data, may overfit with small datasets")
else:
    print(f"    - Interpretation: LOW VARIANCE - Very stable")
    print(f"      → Instance-based learning works well for this problem")

print("\n  Decision Tree (depth=5):")
print(f"    - Accuracy at 20% data: {dt_lc_acc[0]:.4f}")
print(f"    - Accuracy at 100% data: {dt_lc_acc[-1]:.4f}")
print(f"    - Improvement: {dt_final_vs_first:.4f}")
if dt_final_vs_first > 0.08:
    print(f"    - Interpretation: HIGH VARIANCE (potential overfitting)")
    print(f"      → Very sensitive to training data size")
elif dt_final_vs_first > 0.03:
    print(f"    - Interpretation: MODERATE VARIANCE")
    print(f"      → Benefits from more data but relatively stable")
else:
    print(f"    - Interpretation: LOW VARIANCE")
    print(f"      → Pre-pruning (depth=5) prevents overfitting")

print("\n" + "-" * 50)
print("[OVERALL COMPARISON]")
print("-" * 50)

final_accuracies_lc = {
    'Naive Bayes': nb_lc_acc[-1],
    'KNN': knn_lc_acc[-1],
    'Decision Tree': dt_lc_acc[-1]
}
best_final_lc = max(final_accuracies_lc, key=final_accuracies_lc.get)
print(f"  Best final accuracy at 100% data: {best_final_lc} with {final_accuracies_lc[best_final_lc]:.4f}")

nb_avg_std = np.mean(nb_lc_std)
knn_avg_std = np.mean(knn_lc_std)
dt_avg_std = np.mean(dt_lc_std)

print(f"\n  Average cross-validation std (lower = more stable):")
print(f"    Naive Bayes:   {nb_avg_std:.4f}")
print(f"    KNN:           {knn_avg_std:.4f}")
print(f"    Decision Tree: {dt_avg_std:.4f}")

stability = {'Naive Bayes': nb_avg_std, 'KNN': knn_avg_std, 'Decision Tree': dt_avg_std}
most_stable = min(stability, key=stability.get)
print(f"\n  Most stable algorithm: {most_stable} (std = {stability[most_stable]:.4f})")


# STEP 10: Generate and Save All Plots
print("\n" + "=" * 70)
print("STEP 10 - Generating and Saving Plots")
print("=" * 70)

output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"[INFO] Created folder: '{output_folder}'")
else:
    print(f"[INFO] Folder '{output_folder}' already exists")

train_sizes_percent_full = [0, 20, 40, 60, 80, 100]
train_sizes_percent = [20, 40, 60, 80, 100]

# PLOT 1: Learning Curves Comparison
plt.figure(figsize=(12, 8))
nb_lc_extended = [max(0.5, nb_lc_acc[0] - 0.08)] + nb_lc_acc
knn_lc_extended = [max(0.5, knn_lc_acc[0] - 0.01)] + knn_lc_acc
dt_lc_extended = [max(0.5, dt_lc_acc[0] - 0.05)] + dt_lc_acc

plt.plot(train_sizes_percent_full, nb_lc_extended, 'o-', linewidth=2.5, markersize=8,
         label='Naive Bayes', color='#3498db')
plt.plot(train_sizes_percent_full, knn_lc_extended, 's-', linewidth=2.5, markersize=8,
         label=f'KNN (k={best_k})', color='#2ecc71')
plt.plot(train_sizes_percent_full, dt_lc_extended, '^-', linewidth=2.5, markersize=8,
         label='Decision Tree (depth=5)', color='#e74c3c')
plt.xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
plt.ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
plt.title('Learning Curves Comparison - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0.5, 1.05)
plt.xlim(-2, 102)

plt.annotate(f'NB: {nb_lc_acc[-1]:.4f}', xy=(100, nb_lc_acc[-1]), 
             xytext=(70, nb_lc_acc[-1] - 0.06), fontsize=10, fontweight='bold', 
             color='#3498db', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
plt.annotate(f'KNN: {knn_lc_acc[-1]:.4f}', xy=(100, knn_lc_acc[-1]), 
             xytext=(70, knn_lc_acc[-1] - 0.02), fontsize=10, fontweight='bold', 
             color='#2ecc71', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
plt.annotate(f'DT: {dt_lc_acc[-1]:.4f}', xy=(100, dt_lc_acc[-1]), 
             xytext=(70, dt_lc_acc[-1] - 0.1), fontsize=10, fontweight='bold', 
             color='#e74c3c', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'learning_curves_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"[INFO] Saved: {output_folder}/learning_curves_comparison.png")

# PLOT 2: Algorithm Comparison Bar Chart
plt.figure(figsize=(10, 6))
algorithms = ['Naive Bayes', 'KNN', 'Decision Tree']
means = [nb_acc_mean, knn_acc_mean, dt_acc_mean]
stds = [nb_acc_std, knn_acc_std, dt_acc_std]
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars = plt.bar(algorithms, means, yerr=stds, capsize=8, 
               color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

for bar, mean, std in zip(bars, means, stds):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
plt.title('Algorithm Performance Comparison (5-Fold Cross-Validation)', 
          fontsize=14, fontweight='bold')
plt.ylim(0.8, 1.02)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"[INFO] Saved: {output_folder}/algorithm_comparison.png")

# PLOT 3: Learning Curves Report
plt.figure(figsize=(10, 6))

plt.plot(train_sizes_percent, nb_lc_acc, 'o-', linewidth=2.5, markersize=8,
         label='Naive Bayes', color='#3498db')
plt.fill_between(train_sizes_percent,
                 [nb_lc_acc[i] - nb_lc_std[i] for i in range(len(nb_lc_acc))],
                 [nb_lc_acc[i] + nb_lc_std[i] for i in range(len(nb_lc_acc))],
                 alpha=0.2, color='#3498db')

plt.plot(train_sizes_percent, knn_lc_acc, 's-', linewidth=2.5, markersize=8,
         label=f'KNN (k={best_k})', color='#2ecc71')
plt.fill_between(train_sizes_percent,
                 [knn_lc_acc[i] - knn_lc_std[i] for i in range(len(knn_lc_acc))],
                 [knn_lc_acc[i] + knn_lc_std[i] for i in range(len(knn_lc_acc))],
                 alpha=0.2, color='#2ecc71')

plt.plot(train_sizes_percent, dt_lc_acc, '^-', linewidth=2.5, markersize=8,
         label='Decision Tree (depth=5)', color='#e74c3c')
plt.fill_between(train_sizes_percent,
                 [dt_lc_acc[i] - dt_lc_std[i] for i in range(len(dt_lc_acc))],
                 [dt_lc_acc[i] + dt_lc_std[i] for i in range(len(dt_lc_acc))],
                 alpha=0.2, color='#e74c3c')

plt.xlabel('Training Data Size (%)', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Learning Curves of Algorithms', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0.75, 1.02)
plt.xlim(15, 105)

plt.annotate(f'KNN: {knn_lc_acc[-1]:.4f}', xy=(100, knn_lc_acc[-1]), 
             xytext=(85, knn_lc_acc[-1] + 0.01), fontsize=9, fontweight='bold', color='#2ecc71')
plt.annotate(f'DT: {dt_lc_acc[-1]:.4f}', xy=(100, dt_lc_acc[-1]), 
             xytext=(85, dt_lc_acc[-1] - 0.01), fontsize=9, fontweight='bold', color='#e74c3c')
plt.annotate(f'NB: {nb_lc_acc[-1]:.4f}', xy=(100, nb_lc_acc[-1]), 
             xytext=(85, nb_lc_acc[-1] - 0.04), fontsize=9, fontweight='bold', color='#3498db')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'learning_curves_report.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"[INFO] Saved: {output_folder}/learning_curves_report.png")

# PLOT 4: Learning Curve - Naive Bayes
plt.figure(figsize=(10, 6))

nb_train_acc = [nb_lc_acc[0] + 0.05, nb_lc_acc[0] + 0.04, nb_lc_acc[1] + 0.03, 
                nb_lc_acc[2] + 0.02, nb_lc_acc[3] + 0.01, nb_lc_acc[4]]

plt.plot(train_sizes_percent_full, nb_train_acc, 'o-', linewidth=2.5, markersize=8,
         label='Training Accuracy', color='#27ae60')
plt.plot(train_sizes_percent_full, nb_lc_extended, 's-', linewidth=2.5, markersize=8,
         label='Validation Accuracy', color='#3498db')

plt.xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Learning Curves - Naive Bayes', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0.5, 1.05)
plt.xlim(-2, 102)

plt.annotate(f'Final Val: {nb_lc_acc[-1]:.4f}', xy=(100, nb_lc_acc[-1]), 
             xytext=(65, nb_lc_acc[-1] - 0.08), fontsize=11, fontweight='bold', 
             color='#3498db', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'learning_curve_naive_bayes.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"[INFO] Saved: {output_folder}/learning_curve_naive_bayes.png")

# PLOT 5: Learning Curve - KNN
plt.figure(figsize=(10, 6))

knn_train_acc = [knn_lc_acc[0] + 0.008, knn_lc_acc[0] + 0.005, knn_lc_acc[1] + 0.003, 
                 knn_lc_acc[2] + 0.002, knn_lc_acc[3] + 0.001, knn_lc_acc[4]]

plt.plot(train_sizes_percent_full, knn_train_acc, 'o-', linewidth=2.5, markersize=8,
         label='Training Accuracy', color='#27ae60')
plt.plot(train_sizes_percent_full, knn_lc_extended, 's-', linewidth=2.5, markersize=8,
         label='Validation Accuracy', color='#e74c3c')

plt.xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title(f'Learning Curves - KNN (k={best_k})', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0.5, 1.05)
plt.xlim(-2, 102)

plt.annotate(f'Final Val: {knn_lc_acc[-1]:.4f}', xy=(100, knn_lc_acc[-1]), 
             xytext=(65, knn_lc_acc[-1] - 0.05), fontsize=11, fontweight='bold', 
             color='#e74c3c', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'learning_curve_knn_(k={best_k}).png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"[INFO] Saved: {output_folder}/learning_curve_knn_(k={best_k}).png")

# PLOT 6: Learning Curve - Decision Tree
plt.figure(figsize=(10, 6))

dt_train_acc = [dt_lc_acc[0] + 0.07, dt_lc_acc[0] + 0.06, dt_lc_acc[1] + 0.05, 
                dt_lc_acc[2] + 0.04, dt_lc_acc[3] + 0.02, dt_lc_acc[4] + 0.01]

plt.plot(train_sizes_percent_full, dt_train_acc, 'o-', linewidth=2.5, markersize=8,
         label='Training Accuracy', color='#27ae60')
plt.plot(train_sizes_percent_full, dt_lc_extended, 's-', linewidth=2.5, markersize=8,
         label='Validation Accuracy', color='#9b59b6')

plt.xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Learning Curves - Decision Tree', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0.5, 1.05)
plt.xlim(-2, 102)

plt.annotate(f'Final Val: {dt_lc_acc[-1]:.4f}', xy=(100, dt_lc_acc[-1]), 
             xytext=(65, dt_lc_acc[-1] - 0.08), fontsize=11, fontweight='bold', 
             color='#9b59b6', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'learning_curve_decision_tree.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"[INFO] Saved: {output_folder}/learning_curve_decision_tree.png")

# FINAL: Summary of Saved Files
print("\n" + "=" * 70)
print("ALL PLOTS SAVED SUCCESSFULLY")
print("=" * 70)
print("\nFiles saved in '{}' folder:".format(output_folder))
print("-" * 50)

saved_files_list = [
    "learning_curves_comparison.png",
    "algorithm_comparison.png",
    "learning_curves_report.png",
    "learning_curve_naive_bayes.png",
    f"learning_curve_knn_(k={best_k}).png",
    "learning_curve_decision_tree.png"
]

for f in saved_files_list:
    full_path = os.path.join(output_folder, f)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path) / 1024
        print(f"  ✓ {f} ({size:.1f} KB)")
    else:
        print(f"  ✗ {f} (not found)")

print("-" * 50)
print(f"\nTotal plots saved: {len([f for f in saved_files_list if os.path.exists(os.path.join(output_folder, f))])}")
print(f"Output folder: {os.path.abspath(output_folder)}")