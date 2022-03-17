# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:20:31 2022

@author: atlas
"""
import numpy as np
from sklearn.neighbors import KDTree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import random
import time
import trimesh
# %%


def IterativeClosestPoint(source_pts, target_pts, tau=10e-6):
    '''
    inputs:
    source_pts : 3 x N
    target_pts : 3 x M
    tau : threshold for convergence
    Its the threshold when RMSE does not change comapred to the previous 
    RMSE the iterations terminate. 
    outputs:
    R : Rotation Matrtix (3 x 3)
    t : translation vector (3 x 1)
    k : num_iterations
    '''

    k = 0
    current_pts = source_pts.copy()
    last_rmse = 0
    t = np.zeros((3, 1))
    R = np.eye(3, 3)
    errors = []

    # iteration loop
    while True:
        neigh_pts = FindNeighborPoints(current_pts, target_pts)
        (R, t) = RegisterPoints(source_pts, neigh_pts)
        current_pts = ApplyTransformation(source_pts, R, t)
        
        if k % 20 == 0:
            print(k)
            f_out = "Iter_{}.ply".format(str(k))
        trimesh.Trimesh(vertices=current_pts.T).export(f_out)
        
        rmse = ComputeRMSE(current_pts, neigh_pts)
        # print("iteration : {}, rmse : {}".format(k,rmse))
        errors.append(rmse)

        if np.abs(rmse - last_rmse) < tau:
            break
        last_rmse = rmse
        k = k + 1
    f_out2 = "Iter_last.ply"
    trimesh.Trimesh(vertices=current_pts.T).export(f_out2)
    return (R, t, k, errors)


# Computes the root mean square error between two data sets.
# here we dont take mean, instead sum.
def ComputeRMSE(p1, p2):
    return np.sum(np.sqrt(np.sum((p1-p2)**2, axis=0)))


# applies the transformation R,t on pts
def ApplyTransformation(pts, R, t):
    return np.dot(R, pts) + t

# applies the inverse transformation of R,t on pts
def ApplyInvTransformation(pts, R, t):
    return np.dot(R.T,  pts - t)

# calculate naive transformation errors
def CalcTransErrors(R1, t1, R2, t2):
    Re = np.sum(np.abs(R1-R2))
    te = np.sum(np.abs(t1-t2))
    return (Re, te)


# point cloud registration between points p1 and p2
# with 1-1 correspondance
def RegisterPoints(p1, p2):
    u1 = np.mean(p1, axis=1).reshape((3, 1))
    u2 = np.mean(p2, axis=1).reshape((3, 1))
    pp1 = p1 - u1
    pp2 = p2 - u2
    W = np.dot(pp1, pp2.T)
    U, _, Vh = np.linalg.svd(W)
    R = np.dot(U, Vh).T
    if np.linalg.det(R) < 0:
        Vh[2, :] *= -1
        R = np.dot(U, Vh).T
    t = u2 - np.dot(R, u1)
    return (R, t)


# function to find source points neighbors in
# target based on KDTree
def FindNeighborPoints(source, target):
    n = source.shape[1]
    kdt = KDTree(target.T, leaf_size=30, metric='euclidean')
    index = kdt.query(source.T, k=1, return_distance=False).reshape((n,))
    return target[:, index]

# %%
# num random points 
N = 100
X = np.random.rand(3,N) * 100.0

# comment below for using random points
# X = np.loadtxt("bunny.txt")
path = "car_001.obj"
mesh = trimesh.load(path)
X = mesh.vertices
# print(X.shape)
# X = X[::50,0:3].T
X = X[:,0:3].T
# print(X.shape)
N = X.shape[1]

# random rotation and translation
t = np.random.rand(3,1) * 25.0

theta = np.random.rand() * 20
phi = np.random.rand() * 20
psi = np.random.rand() * 20

R = Rot.from_euler('zyx', [theta, phi, psi], degrees = True)

R = R.as_matrix()
# print("Input Rotation : \n{}".format(R))
# print("Input Translation : \n{}".format(t))

# select a subset percentage of points
subset_percent = 40
Ns = int(N * (subset_percent/100.0))
index = random.sample(list(np.arange(N)), Ns)
P = X[:, index]

# apply inverse transformation
P = ApplyInvTransformation(P, R, t)

# ICP algorithm
start = time.time()
Rr, tr, num_iter, errors = IterativeClosestPoint(source_pts = P, target_pts = X, tau = 10e-6)
end = time.time()

print("Time taken for ICP : {}".format(end - start))
print("num_iterations: {}".format(num_iter))
# print("Rotation Estimated : \n{}".format(Rr))
# print("Translation Estimated : \n{}".format(tr))

# calculate error:
Re, te = CalcTransErrors(R, t, Rr, tr)
print("Rotational Error : {}".format(Re))
print("Translational Error : {}".format(te))

# transformed new points
Np = ApplyTransformation(P, Rr, tr)
# %%
# visual errors
fig = plt.figure()
plt.plot(errors)
plt.title("Errors during process")
plt.xlabel("Number of iterations")
plt.ylabel("Error")
plt.show()
'''
# visual output
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[0,:], X[1,:], X[2,:], marker='o', alpha = 0.2, label="input target points")
ax.scatter(P[0,:], P[1,:], P[2,:], marker='^', label="input source points")
ax.scatter(Np[0,:], Np[1,:], Np[2,:], marker='x', label="transformed source points")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
plt.show()
'''