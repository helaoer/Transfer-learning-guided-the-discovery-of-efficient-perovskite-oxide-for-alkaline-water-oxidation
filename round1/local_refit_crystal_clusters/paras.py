import pandas as pd
no_cluster = 7
cluster_centres = []
with open("../cluster/7_crystal_cluster_centers.xlsx", "rb") as f:
    data = pd.read_excel(f)
    cluster_centres = data.values
filepath = "../cluster/overp_crystal_clusters.xlsx"
tuned_parameters = [
    {
        'n_estimators': [3, 5, 10, 20, 50],
        'learning_rate': [0.01, 0.1, 0.5],
        'subsample': [0.5, 0.8, 1.0],
        'max_depth': list(range(2, 10)),
        'min_samples_split': range(2,11,2),
        'min_samples_leaf': range(1,10,2),
        'max_features': [3,5,7],
        'random_state': [1],
    }
]