import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_excel("./3rd_crystal_data.xlsx")

x = data.iloc[:,:-1].values
cluster_no = data.iloc[:,-1].values.reshape((-1))
x_ = x[:,3:-4]
centres = []
for i in range(7):
    x_cluster = x_[cluster_no==i]
    centres.append(np.mean(x_cluster,axis=0))
centres = np.asarray(centres)
fig, ax = plt.subplots(figsize=[10,10])
colors = plt.cm.tab20c(np.linspace(0, 1, 7))
crystal_names = ["unknown","cubic","hexagonal","monoclinic","orthorhombic","rhombohedral","tetragonal"]
for i in range(7):
    cluster_points = x_[cluster_no == i]
    ax.scatter(centres[i, 0], centres[i, 1], marker="x", c='black')
    ax.text(centres[i, 0], centres[i, 1], s="%s"%(crystal_names[i]))
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label='%s: %s'%(crystal_names[i],cluster_points.shape[0]))
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.title("crystal clusters")
plt.legend()
plt.savefig("clusters_crystal.png")
plt.close()

# save to excel
center_df = pd.DataFrame(centres, columns=["center_x_%s"%(i) for i in range(x[0].shape[0]-7)])
center_df.to_excel("7_crystal_cluster_centers.xlsx",index=False) 