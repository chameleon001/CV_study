
#%%
import numpy as np

#%%
def normalize(points):

    for row in points:
        row /= points[-1]
    return points

def make_homog(points):
    return np.vstack((points, np.ones((1,points.shape[1]))))

#%%

# direct linear transformation
def H_from_points(fp,tp):

#  Find homography H, such that fp is mapped to tp using the 
#  linear DLT method. Points are conditioned automatically.

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

#  condition points 
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(tp[:2], axis=1)) + 1e-9

    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd

    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(tp[:2], axis=1)) + 1e-9

    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = np.dot(C2,tp)

    nbr_correspondences = fp.shape[1]
    A = np.zeros((2*nbr_correspondences, 9))

    for i in range(nbr_correspondences):
        A[2*i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i]*tp[0][i], tp[0][i]*tp[1][i], tp[0][i]]
        A[2*i+1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i]*tp[0][i], tp[1][i]*fp[1][i], tp[1][i]]

    # 특이값 분해
    U,S,V = np.linalg.svd(A)
    H = V[8].reshape((3,3))

    # linalg.inv 역행렬.
    H = np.dot(np.linalg.inv(C2), np.dot(H,C1))

    return H/H[2,2]

def Haffine_from_points(fp, tp):
    # Find H, affine transformation, such that
    # tp is affine transf of tp

    if fp.shape != tp.shape:
        raise RuntimeError('numer of points do not match')

    # condition points
    # from points
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    
    C1 = np.diag([1/maxstd, 1/maxstd,1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = np.dot(C1,fp)

    # to points
    m = np.mean

#%%
