
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
def H_from_points(fp,tp):

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    