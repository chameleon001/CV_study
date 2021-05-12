#%%
# from numpy import arange, loadtxt, pi, savetxt
import numpy as np
# from ... import my_Util

import os
import matplotlib.pyplot as plt
from PIL import Image
from numpy import linalg, zeros

public_data_path = "../../data"

# %%
## SIFT

def process_iamge(image_name, result_name, params= '--edge-thresh10--peak-thresh5'):
    # process an image and save the result in a file

    if image_name[-3:] != 'pgm':
        #create a pgm file
        im = Image.open(image_name).convert('L')
        im.save(result_name)
        image_name = result_name

    cmmd = str("sift" + image_name + " --output="+ result_name+" " + params)
    os.system(cmmd)

    print("processed {0}  to  {1}".format(image_name,result_name))

def read_features_from_file(filename):
    """read feature properties and return in matrix"""

    f = np.loadtxt(filename)
    return f[:,:4],f[:,4:]

def write_features_to_file(file_name, locs, desc):
    """save feature location and descriptor to file"""
    np.savetxt(filter, np.hstack((locs,desc)))

def plot_features(im, locs, circle=False):
    """
    locs (row, col, scale, orientation of each feature)
    """

    def draw_circle(c,r):
        t = np.arange(0,1.01,.01)*2*np.pi
        x = r*np.cos(t) + c[0]
        y = r*np.sin(t) + c[1]

        plt.plot(x,y,'b',linewidth=2)
    
    plt.imshow(im)

    if circle:
        for p in locs:
            draw_circle(p[:2],p[2]) 
    else:
        plt.plot(locs[:,0],locs[:,1],'ob')
    np.axis('off')

#%%

image_name = '/empire.jpg'

public = public_data_path + image_name
print(public)
im1 = np.array(Image.open(public).convert('L'))

#이거 안되는데.. unkonwn file extension error
# pgm을 쓰면 디코더 에러나옴
process_name = 'empire.sift'
process_iamge(public, process_name)
l1, d1 = read_features_from_file(process_name)

plt.figure()
plt.gray()
plot_features(im1, l1, circle=True)
plt.show()
# %%
def match(desc1, desc2):
    # desc1 : descriptors for first image
    # desc2 second image (descriptor)

    desc1 = np.array([d/linalg.norm(d) for d in desc1])
    desc2 = np.array([d/linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = zeros((desc1_size[0],1),'int')
    desc2t = desc2.T

    for i in range(desc1_size[0]):
        dotprods = dot(desc1[i,:],desc2t)
        dotprods = 0.9999*dotprods

        indx = np.argsort(np.arccos(dotprods))

        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores