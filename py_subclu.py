import sklearn
import sklearn.cluster, sklearn.datasets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import umap
import sklearn.preprocessing
import pandas as pd
from tqdm.auto import tqdm
import math
from scipy.special import gamma
import umap
import scipy

def generate_dbscan_clusters(X, eps, m):
    """
    Generate clusters on a dataframe X using DBSCAN
    
    Parameters
    ----------
    eps : float
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    """
    
    C = []
    dbscan = sklearn.cluster.DBSCAN(eps=eps,min_samples=m).fit(X)
    for label in np.unique(dbscan.labels_):
        if label!=-1: # -1 means sample is not in a cluster
            C.append(X.iloc[np.where(dbscan.labels_==label)])
    return C

def generate_1D_clusters(X, eps, m):
    """
    Generate all 1-D clusters
    """
    
    C = []
    S = set()
    for a in X.columns:
        C1 = generate_dbscan_clusters(X.loc[:,[a]], eps=eps, m=m)
        for clust in C1:
            C.append(clust)
            S.add(frozenset({a})) # frozenset because sets are mutable (thus non hashable)
    return C, S

def generate_candidates(Sk):
    """
    Generate (k+1)-Dimensional candidates from k-dimensional subspaces
    by merging subspaces sharing k-1 common attributes 
    and pruning irrelevant candidates
    Irrelevant candidates are candidates that contains k-dimensional subspaces
    that are not included in Sk
    """
    
    cands = set()
    for s1 in Sk:
        for s2 in Sk:
            if s1!=s2:
                if len(s1.intersection(s2))==(len(s1)-1): # if s1 and s2 share k-1 features
                    cands.add(s1.union(s2))
    #Prune irrelevant candidates
    to_prune = set()
    for cand in cands:
        for a in cand:
            s = cand.difference(set({a}))
            if not s in Sk:
                print("pruning candidate : ",cand)
                to_prune.add(cand)
                break
    cands -= to_prune
    return cands

def generate_clusters(X, C, Sk, cands, eps, m):
    """
    Generate (k+1)-dimensional clusters (C_k+1) from :
     - k-dimensional clusters (C_k)
     - (k+1)-dimensional candidate subspaces (cands)
     - k-dimensional subspaces
    """
    
    cl_cand = [] # Ck+1
    s_cands = set() # Sk+1
    for cand in (pbar := tqdm(cands)):
        bestS=list(cand)[0]
        minObj = np.inf
        
        for a in cand:  
            s = cand.difference(set({a}))  #Every subspace of cand of dim k is part of Sk (thanks to the pruning step)
            #count the number of objects in this k-dim subspace of that candidate
            nObjInCS = 0
            for clust in C:
                if set(clust.columns)==s:
                    nObjInCS+=clust.shape[0]
            if nObjInCS<minObj:
                bestS = s
                minObj = nObjInCS
        
        #print("Best subspace : ", s)
        
        for clust in C:
            if set(clust.columns)==bestS: #clust in best subspace
                cl_dbscan_cand = generate_dbscan_clusters(X.loc[clust.index,cand], eps, m)
                for cl in cl_dbscan_cand:
                    cl_cand.append(cl)
                    s_cands.add(frozenset(cl.columns))
                    
    return cl_cand, s_cands

def SUBCLU(X, eps, m):
    C1, S1 = generate_1D_clusters(X,eps,m)
    
    C = [C1]
    S = [S1]
    
    while len(C[-1])>0:
        print("Computing {}-dim clusters".format(len(C)+1))
        cands = generate_candidates(S[-1])
        Ck, Sk = generate_clusters(X, C[-1], S[-1], cands, eps, m)
        C.append(Ck)
        S.append(Sk)
    return C, S

def qualityS2(X, S, eps, m):
    """
    Computes the quality of a subspace (no need of its clusters)
    Formula written in the RIS paper (Kailing 2003)
    """
    
    neighbors = np.sum(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X.loc[:,S]))<=eps,axis=0) #1D-array with the number of neighbours of each point in S
    countS = neighbors[neighbors>=m].sum() #Sum of all neighbors in neighborhood of core objects
    d=len(S)
    return countS / (X.shape[0]**2 * ((math.sqrt(math.pi**d) / gamma(d/2 + 1)) * (eps**d)))
    
def allSubspacesQuality(X, S, eps, m):
    dictSubspacesQuality = dict()
    for s in S:
        dictSubspacesQuality[s] = qualityS2(X, s, eps, m)
    return dictSubspacesQuality

def clusters_list_to_dataframe(C,Q):
    """
    Create a dataframe to store all relevant informations about a cluster and its corresponding subspace
    """
    
    df = pd.DataFrame(columns=["Cluster","Number of instances","Subspace","Subspace quality","Subspace size","Indices"])
    C = [cl for sublist in C for cl in sublist]
    for i,cl in enumerate(C):
        df.loc[i,"Cluster"]=cl.values
        df.loc[i,"Number of instances"]=cl.shape[0]
        df.loc[i,"Subspace size"]=cl.shape[1]
        df.loc[i,"Subspace"]=set(cl.columns)
        df.loc[i,"Subspace quality"]=Q[frozenset(cl.columns)]
        df.loc[i,"Indices"]=cl.index
    return df

def draw_subspaces_clusters(X,df_C,limit=None):
    """
    Draw a figure for each subspace, coloring each dataset
    Using UMAP for subspaces of dimension > 2
    """
    
    subspaces_ordered_list = df_C.sort_values("Subspace quality",ascending=False)["Subspace"].drop_duplicates()
    X_scaled = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(X),columns=X.columns)
    
    if not limit:
        limit=subspaces_ordered_list.shape[0]
    for s in subspaces_ordered_list[:limit]:
        fig = plt.figure(figsize=(9,4))
        if len(s)==1: ## 1-Dim subspaces
            rand_jitter = (np.random.rand(X.shape[0])-0.5)*0.2
            plt.ylim([-1,1])
            plt.xlabel(list(s)[0])
            plt.ylabel("Jitter")
            plt.scatter(x=X.loc[:,s],y=rand_jitter)
            for ic in df_C.loc[df_C["Subspace"]==s,"Indices"]:
                plt.scatter(x=X.loc[ic,s],y=rand_jitter[ic])
        
        elif len(s)==2: ## 2-Dim subspaces
            labels = list(s)
            x_label = labels[0]
            y_label = labels[1]
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.scatter(x=X.loc[:,x_label],y=X.loc[:,y_label])
            for ic in df_C.loc[df_C["Subspace"]==s,"Indices"]:
                plt.scatter(x=X.loc[ic,x_label],y=X.loc[ic,y_label])
        
        elif len(s)==3: ## 3-dim subspaces
            plt.gcf().set_size_inches(18.5,10.5)
            ax = fig.add_subplot(1,2,1,projection="3d")
            labels = list(s)
            x_label = labels[0]
            y_label = labels[1]
            z_label = labels[2]
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            total_ic=[]
            for i,ic in enumerate(df_C.loc[df_C["Subspace"]==s,"Indices"]):
                ax.scatter(xs=X.loc[ic,x_label],ys=X.loc[ic,y_label],zs=X.loc[ic,z_label],c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i+1])
                total_ic.append(ic)
            nc = np.setdiff1d(X.index,np.array([ic for sublist in total_ic for ic in sublist]))
            ax.scatter(xs=X.loc[nc,x_label],ys=X.loc[nc,y_label],zs=X.loc[nc,z_label])
            
            ax2 = fig.add_subplot(1,2,2)
            reducer = umap.UMAP().fit(X_scaled.loc[:,s])
            ax2.set_xlabel("UMAP X")
            ax2.set_ylabel("UMAP Y")
            ax2.scatter(x=reducer.embedding_[:,0],y=reducer.embedding_[:,1])
            for ic in df_C.loc[df_C["Subspace"]==s,"Indices"]:
                ax2.scatter(x=reducer.embedding_[ic,0],y=reducer.embedding_[ic,1])
                
        else : ## >3-dim subspaces
            reducer = umap.UMAP().fit(X_scaled.loc[:,s])
            plt.xlabel("UMAP X")
            plt.ylabel("UMAP Y")
            plt.scatter(x=reducer.embedding_[:,0],y=reducer.embedding_[:,1])
            for ic in df_C.loc[df_C["Subspace"]==s,"Indices"]:
                plt.scatter(x=reducer.embedding_[ic,0],y=reducer.embedding_[ic,1])
        plt.title("Subspace " + str(s) + " with quality "+ str(df_C.loc[df_C["Subspace"]==s,"Subspace quality"].mean()))
        plt.show()