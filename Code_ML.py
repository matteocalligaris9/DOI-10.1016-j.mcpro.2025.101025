#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay, roc_curve
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score
from hyperopt.pyll import scope
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

#Import the df normalized in R.
#Condition is in 'Condition' column
df = pd.read_csv('normalized_sig_proteins.csv', sep=';', index_col=0)

#Fill Na with 0
df=df.fillna(0)

#Prepare matrix for ML
X = df.drop(columns=['Condition'])
y=df.Condition
y = pd.factorize(y)

#Prepare hyperparameter
search_spaces = {
    "KNN": {
        "n_neighbors": scope.int(hp.quniform("n_neighbors", 3, 15,1)),
        "weights": hp.choice("weights", ["uniform","distance"]),
        "p": scope.int(hp.quniform("p", 1, 3, 1)),
    },
    "SVM": {
        "C": hp.loguniform("C", np.log(0.01), np.log(100)),
        "kernel": hp.choice("kernel", ["linear", "rbf", "poly"]),
        "gamma": hp.loguniform("gamma", np.log(1e-4), np.log(1)),
        "degree": hp.choice("degree", [2, 3, 4, 5]),
    },
    "Random Forest": {
        "n_estimators":scope.int(hp.quniform("n_estimators", 2, 20, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 2, 30, 1)),
        "min_samples_split": hp.uniform("min_samples_split", 0.01, 0.5),
        "min_samples_leaf": hp.uniform("min_samples_leaf", 0.01, 0.5),
        "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
    },
    "XGBoost": {
        "n_estimators":scope.int(hp.quniform("n_estimators", 2, 20, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 2, 10, 1)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "gamma": hp.loguniform("gamma", np.log(1e-4), np.log(1)),
    },
}

#Divide the df in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y[0], test_size=0.3, stratify=y[0], random_state=42)

#Prepare the compilation of the hyperparameter
def objective_knn(params):
    model = KNeighborsClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    return {"loss": 1 - auc, "status": STATUS_OK}

def objective_svm(params):
    model = SVC(probability=True, decision_function_shape="ovr", **params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    return {"loss": 1 - auc, "status": STATUS_OK}

def objective_rf(params):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    return {"loss": 1 - auc, "status": STATUS_OK}

def objective_xgb(params):
    model = XGBClassifier(eval_metric="mlogloss", objective="multi:softprob", num_class=3, **params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    return {"loss": 1 - auc, "status": STATUS_OK}

#Run the compilation
choice_maps = {
    "KNN": {
        "weights": ["uniform", "distance"],
    },
    "SVM": {
        "kernel": ["linear", "rbf", "poly"],
        "degree": [2, 3, 4, 5],
    },
    "Random Forest": {
        "max_features": ["sqrt", "log2", None],
    },
}

objective_funcs = {
    "KNN": objective_knn,
    "SVM": objective_svm,
    "Random Forest": objective_rf,
    "XGBoost": objective_xgb,
}

best_hyperparams = {}
for model_name, space in search_spaces.items():
    print(f"Optimizing {model_name}...")
    trials = Trials()
    best = fmin(
        fn=objective_funcs[model_name],
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
    )
    # Map choices from indices to actual values
    if model_name in choice_maps:
        for param, options in choice_maps[model_name].items():
            if param in best:
                best[param] = options[best[param]]

    best_hyperparams[model_name] = best
    print(f"Best hyperparameters for {model_name}: {best}")

#Extract the best model
final_models = {}
for model_name, params in best_hyperparams.items():
    if model_name == "KNN":
        params["n_neighbors"] = int(params["n_neighbors"])
        model = KNeighborsClassifier(**params)

    elif model_name == "SVM":
        model = SVC(probability=True, decision_function_shape="ovr", **params)

    elif model_name == "Random Forest":
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])
        model = RandomForestClassifier(**params)

    elif model_name == "XGBoost":
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])
        model = XGBClassifier(eval_metric="logloss", objective="multi:softprob", num_class=3, **params)

    # ✅ This line should apply to ALL models
    model.fit(X_train, y_train)
    final_models[model_name] = model

#Plot the ROC curve and Confusion matrix for each model

for name, model in final_models.items():
    model.fit(X_train, y_train)
    # Binarize true labels
    # Define custom class names (ensure they correspond to your class order)
    class_names = ['CT', 'AD TAU-', 'AD TAU+']

    # Binarize y_test based on model.classes_
    classes = model.classes_  # e.g., array([0, 1, 2]) or ['CT', 'AD TAU-', 'AD TAU+']
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)

    # Plot
    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(class_names):
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i],
            y_score[:, i],
            name=class_label,
            ax=plt.gca()
        )

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Multiclass ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Astral/' + name + '_ROC_Astral.png', dpi=300)
    plt.show()

    y_true = y_test
    y_pred = model.predict(X_test)

    accuracy = round(accuracy_score(y_true, y_pred), 3)
    recall = round(recall_score(y_true, y_pred, average='weighted'), 3)
    precision = round(precision_score(y_true, y_pred, average='weighted'), 3)

    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('Precision: ', precision)
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=["Control", "AD TAU -", "AD TAU +"], columns=["Control", "AD TAU -", "AD TAU +"])
    plt.figure()
    chart = sns.heatmap(df_cm, annot=True, fmt=".0f", cmap=cmap, annot_kws={"fontsize": 14})
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
    chart.set_yticklabels(chart.get_yticklabels(), rotation=45)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(name, fontsize=14)
    plt.savefig('Astral/' + name + '_cm_Astral.png', bbox_inches="tight", dpi=600)
    plt.show()

# Assume your best model is:
best_xgb_model = final_models["XGBoost"]

# Compute permutation importance
from sklearn.inspection import permutation_importance
result = permutation_importance(best_xgb_model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)

# Wrap in a DataFrame
perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': result.importances_mean,
    'STD': result.importances_std
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=perm_df.head(10), orient="h")
plt.title("Top 10 Permutation Importances - XGBoost")
plt.tight_layout()
plt.savefig('Astral/XGBoost_Importance_Astral.png', dpi=600)
plt.show()


#Isolate the SHAP values of the test set for XGBoost and plot them

explainer = shap.Explainer(final_models['XGBoost'])
shap_values = explainer(X_test)

n_classes = shap_values.shape[-1]

for i in range(n_classes):
    print(f"Saving SHAP beeswarm for class {i}")

    shap.summary_plot(shap_values[..., i], X_test, plot_type="dot", show=False)
    plt.gcf().savefig(f"Astral/shap_beeswarm_class_{i}_xgb.png", bbox_inches="tight", dpi=300)
    plt.close()


