#%%
import numpy as np
from numpy import linalg
from numpy.lib.npyio import loadtxt
from numpy.random import random
from scipy import ndimage
import matplotlib.pylab as plt
from PIL import Image
import scipy
import Homographies

# transformed_im = ndimage.affine_transform(im, A, b, size)

im =np.array(Image.open('../data/empire.jpg').convert('L'))
H = np.array([[1.4, 0.05, -100], [0.05, 1.5, -100], [0,0,1]])
im2 = ndimage.affine_transform(im, H[:2,:2], (H[0,2], H[1,2]))

plt.figure()
plt.gray()
plt.imshow(im2)
plt.show()
# %%
def image_in_image(im1, im2, tp):

    """
    Put im1 in im2 with an affine transformation
    such that corners are as close to tp as possible
    tp are homogeneous and counter-clockwise from top left 
    """

    m,n = im1.shape[:2]
    fp = np.array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

    H = Homographies.Haffine_from_points(tp,fp)

    im1_t = ndimage.affine_transform(im1,H[:2,:2], (H[0,2],H[1,2]), im2.shape[:2])

    alpha = (im1_t > 0)

    return (1-alpha)*im2 + alpha*im1_t

#%%
def alpha_for_triangle(points,m,n):
    """
    creates alpha map of size(m,n)
    for a triangle with corners defined by points
    """
    alpha = np.zeros((m,n))

    for i in range(int(min(points[0])), int(max(points[0]))):
        for j in range(int(min(points[1])), int(max(points[1]))):
            x = np.linalg.solve(points, [i,j,1])
            if min(x) >0:
                alpha[i,j] =1
    return alpha

# %%

# im1 = np.array(Image.open('../data/beatles.jpg').convert('L'))
# im2 = np.array(Image.open('../data/billboard_for_rent.jpg').convert('L'))

im1 = np.array(Image.open('../data/book_perspective.jpg').convert('L'))
im2 = np.array(Image.open('../data/fisherman.jpg').convert('L'))

tp = np.array([[264,538,540,264],[40,36,605,605],[1,1,1,1]])
# tp = np.array([[675,826,826,677],[55,52,281,277],[1,1,1,1]])

im3 = image_in_image(im1,im2,tp)

plt.figure()
plt.gray()
plt.imshow(im3)
plt.axis('equal')
plt.axis('off')
plt.show()
# %%

m,n = im1.shape[:2]
fp = np.array([[0,m,m,0], [0,0,n,n], [1,1,1,1]])

tp2 = tp[:,:3]
fp2 = fp[:,:3]

H = Homographies.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1, H[:2,:2], (H[0,2],H[1,2]), im2.shape[:2])

alpha = alpha_for_triangle(tp2, im2.shape[0], im2.shape[1])
im3 = (1-alpha)*im2 + alpha*im1_t

tp2 = tp[:,[0,2,3]]
fp2 = fp[:,[0,2,3]]

H = Homographies.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1, H[:2,:2], (H[0,2], H[1,2]), im2.shape[:2])

alpha = alpha_for_triangle(tp2, im2.shape[0], im2.shape[1])
im4 = (1-alpha)*im3 + alpha*im1_t

plt.figure()
plt.gray()
plt.imshow(im4)
plt.axis('equal')
plt.axis('off')
plt.show()
# %%

# import matplotlib.delaunay as md
from scipy.spatial import Delaunay
#matplotlib 에서 scipy spatial Delaunay로 바뀜.

x,y = np.array(np.random.standard_normal((2,100)))
# centers, edges, tri, neighbors = Delaunay(x,y)
tri= Delaunay(np.c_[x,y,]).simplices

plt.figure()

for t in tri:
    t_ext = [t[0], t[1], t[2], t[0]] 
    plt.plot(x[t_ext], y[t_ext], 'r')

plt.plot(x,y,'*')
plt.axis('off')
plt.show()
# %%

def triangulate_points(x,y):
    tri= Delaunay(np.c_[x,y,]).simplices

    return tri
# %%
def pw_affine(fromim, toim, fp, tp, tri):
    """
    Warp triangular patches from an image
    fromim = image to warp
    toim = destination image
    fp = from points in hom. coordinates
    tp = to points in hom. coordinates
    tri = triangulation.
    """

    im = toim.copy()

    #check if image is grayscale or color
    is_color = len(fromim.shape) == 3

    #create image to warp to
    im_t = np.zeros(im.shape, 'uint8')

    for t in tri:
        H = Homographies.Haffine_from_points(tp[:,t], fp[:,t])

        if is_color:
            for col in range(fromim.shape[2]):
                im_t[:,:,col] = ndimage.affine_transform(fromim[:,:,col], H[:2,:2], (H[0,2], H[1,2]), im.shape[:2])
        else:
            im_t = ndimage.affine_transform(fromim,H[:2,:2], (H[0,2], H[1,2]), im.shape[:2])

    # alpha for triangle
    alpha = alpha_for_triangle(tp[:,t], im.shape[0], im.shape[1])

    im[alpha>0] = im_t[alpha>0]

    return im
# %%
def plot_mesh(x,y,tri):

    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]]
        plt.plot(x[t_ext], y[t_ext], 'r')
#%%

fromim = np.array(Image.open('../data/sunset_tree.jpg'))
x,y = np.meshgrid(range(5), range(6))

x = (fromim.shape[1]/4 * x.flatten())
y = (fromim.shape[0]/5 * y.flatten())

tri = triangulate_points(x,y)

im = np.array(Image.open('../data/turningtorso1.jpg'))
tp = loadtxt('../data/turningtorso1_points.txt')

fp = np.vstack((y,x,np.ones((1,len(x)))))
tp = np.vstack((tp[:,1], tp[:,0], np.ones((1,len(tp)))))

im = pw_affine(fromim, im, fp, tp, tri)

plt.figure()
plt.imshow(im)
plot_mesh(tp[1], tp[0], tri)
plt.axis('off')
plt.show()
# %%

from xml.dom import minidom

def read_points_from_xml(xmlFileName):

    xmldoc = minidom.parse(xmlFileName)
    facelist = xmldoc.getElementsByTagName('face')
    faces = {}

    for xmlFace in facelist:
        fileName = xmlFace.attributes['file'].value
        xf = int(xmlFace.attributes['xf'].value)
        yf = int(xmlFace.attributes['yf'].value)
        xs = int(xmlFace.attributes['xs'].value)
        ys = int(xmlFace.attributes['ys'].value)
        xm = int(xmlFace.attributes['xm'].value)
        ym = int(xmlFace.attributes['ym'].value)

        faces[fileName] = np.array([xf,yf,xs,ys,xm,ym])

    return faces


# %%

from scipy import linalg

def compute_rigid_transform(refpoints, points):

    print(points)
    A = np.array([ 
                [points[0], -points[1], 1, 0],
                [points[1], points[0], 0, 1],
                [points[2], -points[3], 1, 0],
                [points[3], points[2], 0, 1],
                [points[4], -points[5], 1, 0],
                [points[5], points[4], 0, 1]
                ])

    y = np.array([refpoints[0],
    refpoints[1],
    refpoints[2],
    refpoints[3],
    refpoints[4],
    refpoints[5]])

    a,b,tx,ty = linalg.lstsq(A,y)[0]
    R = np.array([[a,-b], [b,a]])

    return R, tx, ty
# %%
from scipy import ndimage
import scipy
import os

def rigid_alignment(faces, path, plotflag=False):
    """
    Align images rigidly and save as new images.
    path determines where the aligned images are saved
    set plotflag = True to plot the images.
    """

    # take the points in the first image as refer points

    #책은 왜 항상 안될까..
    # refpoints = faces.values()[0]
    values_view = faces.values()
    value_iter = iter(values_view)
    refpoints = next(value_iter)
    print(refpoints)
    for face in faces:
        points = faces[face]

        R, tx, ty = compute_rigid_transform(refpoints, points)
        T = np.array([[R[1][1], R[1][0], R[0][1], R[0][0]]])

        im = np.array(Image.open(os.path.join(path, face)))
        im2 = np.zeros(im.shape, 'uint8')

        for i in range(len(im.shape)):
            im2[:,:,i] = ndimage.affine_transform(im[:,:,i],linalg.inv(T),offset=[-ty,-tx])
        
        if plotflag:
            plt.imshow(im2)
            plt.show()

        h,w = im2.shape[:2]
        border = (w+h)/20

        scipy.imsave(os.path.join(path,'aligned/'+face),im2[border:h-border,border:w-border,:])
# %%

xmlFileName = '../data/jkfaces.xml'
points = read_points_from_xml(xmlFileName)

rigid_alignment(points, '../data/jkfaces/')
# %%
