import torch
import numpy as np
from lib.controller import vanilla
from lib.gramians import Make
from lib.controller import vanilla, decompose
import json

def ou_process(T,sig2,myseed): 
    # Ornstein-Uhlenbeck (OU)process  
    np.random.seed(seed=myseed)
    n=200
    tau_ = 50
    ou = torch.zeros(T,8,n)
    ou[0] = torch.randn(8,n) * np.sqrt(sig2)
    for t in range(T-1):
        ou[t+1] = ou[t] + (-ou[t]/ tau_ + np.sqrt(2 / tau_ * sig2)  * torch.randn(8,n))                     
    return ou

# def min_max(fr):
#     t,c,n = fr.shape
#     bigA = torch.flatten(fr.transpose(0,1),0,1)
#     ranges = torch.max(bigA, 0)[0] - torch.min(bigA, 0)[0]
#     normFactors = ranges + 5
#     bigA = (bigA - torch.min(bigA, 0)[0])/ normFactors[None, :]
#     x = bigA.reshape(c,t,n)
#     x=x.transpose(0,1)
#     return x


def reduct(fr, dim, var=False, soft = 5): 
    seed = 2023
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed) 
    t,c,n = fr.shape
    bigA = torch.flatten(fr.transpose(0,1),0,1)
    ranges = torch.max(bigA, 0)[0] - torch.min(bigA, 0)[0]
    normFactors = ranges + soft
    bigA = bigA / normFactors[None, :]

    sumA = torch.sum(fr,1)/ normFactors[None, :]        
    meanA = sumA / c

    bigA = bigA - torch.tile(meanA, (c, 1))  
    
    if var:
        from sklearn.decomposition import PCA
        pca = PCA()
        bigA = bigA.numpy()
        rawScores = pca.fit_transform(bigA)
        Scores = rawScores[:,:dim]
        explained_variances = pca.explained_variance_ratio_   
        return Scores, explained_variances[:dim]        
    else:        
        u,_,_ = torch.pca_lowrank(bigA, q=dim)
        return u

def normalize(fr): 
    seed = 2023
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed) 
    t,c,n = fr.shape
    bigA = torch.flatten(fr.transpose(0,1),0,1)
    ranges = torch.max(bigA, 0)[0] - torch.min(bigA, 0)[0]
    normFactors = ranges 
    bigA = bigA / normFactors[None, :]

    sumA = torch.sum(fr,1)/ normFactors[None, :]        
    meanA = sumA / c

    bigA = bigA - torch.tile(meanA, (c, 1))    
    return bigA         


def cca_analysis(X,Y,dim):
    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components=dim)    
    cca.fit(X,Y)
    X_c, Y_c = cca.transform(X,Y) 
    r = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]     
    return X_c, Y_c, r


def q_mov(a,c):
    G = Make(a = a, gamma = c)
    q = G.OSub.m_norm
    n = a.shape[0]
#     q_ = 0.2*q + 0.8*n*c.T@c / np.trace(c.T@c)
    q_ = q
    # Symmetrize for numerical stability
    return 0.5 *(q_ + q_.T)   

def loop_params(w,c,r):
    n = w.shape[0]
    a = w - np.eye(n)
    q = q_mov(a,c) 
    yx_xy = vanilla(a, r, q)
    xz, zy, yx = decompose(n, yx_xy, reg=0.1)
    return xz, zy, yx

