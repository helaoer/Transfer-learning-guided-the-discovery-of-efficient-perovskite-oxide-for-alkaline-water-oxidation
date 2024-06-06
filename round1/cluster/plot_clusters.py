import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

range_of_clusters = range(1,20)
# Load the data
data = pd.read_excel("../encode/data/data.overp.encoded.xlsx")

x = data.iloc[:,3:-1].values
y = data.iloc[:,-1].values.reshape((-1,1))
abo3 = data.iloc[:,:3].values

data_knowledge = pd.read_excel("./data_initial_cluster.xlsx")
true_cluster = data_knowledge.iloc[:,-1].values.reshape((-1))

dists = []
# comparing with data_knowledge
results = []
for n_clusters in range_of_clusters:
    print("n_clusters: ",n_clusters)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    # embedding only
    #######
    kmeans.fit(x[:,:-3])
    labels = kmeans.predict(x[:,:-3])
    #######
    centers = kmeans.cluster_centers_
    dists.append(kmeans.inertia_)
    if n_clusters > 1:
        results.append(
            [
                n_clusters,
                metrics.homogeneity_score(true_cluster, labels),
                metrics.completeness_score(true_cluster, labels), 
                metrics.v_measure_score(true_cluster, labels),
                metrics.adjusted_rand_score(true_cluster, labels),
                metrics.silhouette_score(x[:,:-3], labels, metric='sqeuclidean'),
            ]
        )
        
    # plot
    fig, ax = plt.subplots(figsize=[10,10])
    colors = plt.cm.tab20c(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        cluster_points = x[labels == i]
        ax.scatter(centers[i, 0], centers[i, 1], marker="x", c='black')
        ax.text(centers[i, 0], centers[i, 1], s="cluster %s"%(i))
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label='Cluster %s: %s'%(i,cluster_points.shape[0]))
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("%s clusters"%(n_clusters))
    plt.legend()
    plt.savefig("clusters_%s.png"%(n_clusters))
    plt.close()

    # save to excel
    labels = np.asarray(labels).reshape((-1,1))
    columns = ["ABO3","v_sum","valences",]
    for i in range(x[0].shape[0]-3):
        columns.append("x_%s"%(i))
    columns = columns + ["loading","electrolyte","substrate","overpotential (mV)","cluster_no"]
    df = pd.DataFrame(np.concatenate((abo3,x,y,labels), axis=1),columns=columns)
    df.to_excel("./overp_%s_clusters.xlsx"%(n_clusters),index=False) 
    center_df = pd.DataFrame(centers, columns=["center_x_%s"%(i) for i in range(x[0].shape[0]-3)])
    center_df.to_excel("%s_cluster_centers.xlsx"%(n_clusters),index=False) 

results = np.asarray(results)
# comparing knowledge results
df = pd.DataFrame(results,columns=["n_clusters","homogeneity_score","completeness_score","v_measure_score","adjusted_rand_score","silhouette_score"])
df.to_excel("./kmeans_scores.xlsx",index=False) 

plt.figure(figsize=(14, 10))
plt.plot(results[:,0], results[:,1], label="Homogeneity Score")
plt.plot(results[:,0], results[:,2], label="Completeness Score")
plt.plot(results[:,0], results[:,3], label="V-Measure Score")
plt.plot(results[:,0], results[:,4], label="Adjusted Rand Index")
plt.plot(results[:,0], results[:,5], label="Silhouette Coefficient")
plt.legend(loc='best')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Scores by Number of Clusters')
plt.grid()
plt.savefig("Cluster Scores.png")
plt.close()

# dists
df = pd.DataFrame(np.concatenate((np.asarray(range_of_clusters).reshape(-1,1),np.asarray(dists).reshape(-1,1)),axis=1),columns=["n_clusters","dists"])
df.to_excel("./kmeans_dists.xlsx",index=False) 

# Plot the inertia for each number of clusters
fig, ax = plt.subplots(figsize=[10,10])
plt.bar(list(range_of_clusters),dists)
# Calculate percentage decrease
percentages = []
prev_value = dists[0]
for value in dists[1:]:
    # abs 
    decrease = prev_value - value
    percentages.append(decrease)
    prev_value = value
# Adding text for the percentage decrease
for i, perc in enumerate(percentages):
    plt.text(i+2, dists[i+1], f"{perc:.1f}", ha='center', va='bottom')
plt.xlabel("n_clusters")
plt.ylabel("Dists")
plt.title("Mean Dists")
plt.grid()
plt.savefig("Dists_abs.png")
plt.close()


# Plot the inertia for each number of clusters
fig, ax = plt.subplots(figsize=[10,10])
plt.bar(list(range_of_clusters),dists)
# Calculate percentage decrease
percentages = []
prev_value = dists[0]
for value in dists[1:]:
    decrease = (prev_value - value) / prev_value * 100
    percentages.append(decrease)
    prev_value = value
for i, perc in enumerate(percentages):
    plt.text(i+2, dists[i+1], f"{perc:.1f}%", ha='center', va='bottom')
plt.xlabel("n_clusters")
plt.ylabel("Dists")
plt.title("Mean Dists")
plt.grid()
plt.savefig("Dists_percentage.png")
plt.close()