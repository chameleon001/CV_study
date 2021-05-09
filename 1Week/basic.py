#%%
#numpy와 pil관련 라이브러리 간단한 예제 소스.

from PIL import Image
from pylab import *
#%%
public_data_path = "../data"
#%%
## PIL 라이브러리 사용법..

# pil_img = Image.open('../data/empire.jpg')
pil_img = Image.open(public_data_path+'/empire.jpg').convert('L')
# Color conversions are done using theconvert()method

imshow(pil_img)
show()
# %%

import os

def save_imlist(filelist):
    for infile in filelist:
        outfile = os.path.splitext(infile)[0] + ".jpg"
        if infile != outfile:
            try:
                Image.open(infile).save(outfile)
            except IOError:
                print("cannot convert", infile)

# %%
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    
public_data_list = get_imlist(public_data_path)
# %%
pil_img.thumbnail((128,128))

#원하는 region만큼 짜를 수 있음
box = (100,100,400,400)
region = pil_img.crop(box)
imshow(region)
# %%
region = region.transpose(Image.ROTATE_180)
pil_img.paste(region,box)
imshow(pil_img)
# %%
out = pil_img.resize((128,128))
out = pil_img.rotate(45)
# %%

from PIL import Image
from pylab import *

#read image to array
im = array(Image.open(public_data_path+'/empire.jpg'))

#plot the image
imshow(im)

x = [100, 100, 400, 400]
y = [200, 500, 200, 500]

plot(x,y,'r*')
plot(x[:2],y[:2])

title('Plotting : empire.jpg')
show()
# %%

axis('off')

plot(x,y)
plot(x,y,'r*')
plot(x,y,'go-')
plot(x,y,'ks:')

# 'b' :blue
# 'g' :green
# 'r' :red
# 'c' :cyan
# 'm' :magenta
# 'y' :yellow
# 'k' :black
# 'w' :white

# '-' :solid
# '- -' :dashed
# ':' : dotted

# '.' :point
# 'o' :circle
# 's' :square
# '*' :star
# '+' :plus
# 'x' :x
# %%

from PIL import Image
from pylab import *

im = array(Image.open(public_data_path+'/empire.jpg').convert('L'))

figure()
gray()

contour(im, origin='image')
axis('equal')
axis('off')
# %%

figure()
hist(im.flatten(),128)
show()
# %%

from PIL import Image
from pylab import *

im = array(Image.open(public_data_path+'/empire.jpg'))

imshow(im)

print('Please click 3 points')
x = ginput(3)
print('you clicked :',x)
show()
# %%

im = array(Image.open(public_data_path+'/empire.jpg'))
print("imshpae ::{0} , im dtype :: {1}".format(im.shape, im.dtype))

# %%
im = array(Image.open(public_data_path+'/empire.jpg').convert('L'),'f')
print("imshpae ::{0} , im dtype :: {1}".format(im.shape, im.dtype))

# %%

# value = im[i,j,k]
# im[i,:]=im[j,:] # set the values of row i with values from row j 
# im[:,i]=100 #set all values in column i to 100 
# im[:100,:50].sum() #the sum of the values of the first 100 rows and 50 columns 
# im[50:100,50:100] #rows 50-100, columns 50-100 (100th not included) 
# im[i].mean() #average of row i 
# im[:,-1] #last column 
# im[-2,:](orim[-2]) #second to last row
# %%
from PIL import Image
from numpy import *

im = array(Image.open(public_data_path+'/empire.jpg').convert('L'))
im2 = 255
im3 = (100.0/255) * im + 100
im4 = 255.0 * (im/255.0)**2
# %%
print('min :: {0}, max :: {1} '.format(int(im.min()), int(im.max())))
#%%
# %%
pil_img = Image.fromarray(im)

pil_img = Image.fromarray(uint8(im))
# %%
def imresize(im, sz):
    pil_img = Image.fromarray(unint8(im))

    return array(pil_img.resize(sz))
#%%
def histeq(im, nbr_bins=256):
    """ Histogram equalization grayscale image"""

    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum() # cumulative distribution func
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(), bins[:-1],cdf)

    return im2.reshape(im.shape), cdf
# %%
im = array(Image.open(public_data_path+'/AquaTermi_lowcontrast.jpg').convert('L'))
im2,cdf = histeq(im)
# %%

def compute_average(imlist):
    averageim = array(Image.open(imlist[0]),'f')

    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname + '....skipped')
    
    averageim /= len(imlist)

    return array(averageim, 'uint8')

# %%

def pca(X):

    # get dimensions
    num_data, dim = X.shape
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim>num_data:
        M = dot(X,X.T)
        e, EV = linalg.eigh(M)
        tmp = dot(X.T,EV)
        V = tm[::-1]
        S = sqrt(e)[::-1]

        for i in range(V.shape[1]):
            V[:,i] /=S
    else:
        U,S,A = linalg.svd(X)
        V = V[:num_data]

    return V,S,mean_X
# %%
from PIL import Image
from numpy import * 
from pylab import * 
#%%

im = array(Image.open(public_data_list[0]))
m,n = im.shape[0:2]
imnbr = len(public_data_list)
print("what is :: {0}".format(public_data_list[0]))

# test = [array(Image.open(im)).flatten() for im in public_data_list]

immatrix=np.array([array(Image.open(im)).flatten()for im in public_data_list],'f')

#immatrix = [array(Image.open(im),'f').flatten() for im in public_data_list]

#V,S, immean = pca.pca(immatrix)
V,S, immean = pca(immatrix)


figuxre()
gray()
subplot(2,4,1)
imshow(immean.reshape(m,n))

for i in range(7):
    subplot(2,4,i+2)
    imshow(V[i].reshape(m,n))

show()
# %%

import pickle

f = open('font_pca_modes.pkl','wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()
# %%
f=open('font_pca_modes.pkl','rb')
immean=pickle.load(f)
V=pickle.load(f)
f.close()
# %%
with open('font_pca_modes.pkl', 'wb') as f:
    pickle.dump(immean,f)
    pickle.dump(V,f)
# %%
with open('font_pca_modes.pkl', 'rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)

savetxt('test.txt',x,'%i')
x = loadtxt('test.txt')
#%%
# import PIL.Image as pilimg
# # from PIL import Image
# import numpy as np
# # from numpy import * 
# from scipy.ndimage import filters
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from numpy import *

im = array(Image.open(public_data_path+'/empire.jpg').convert('L'))
im_shape = im.shape
# im2 = filters.gaussian_filter(im,5)
# im = mpimg.imread(public_data_path+'/empire.jpg')
im2 = filters.gaussian_filter(im,5)
# %%

im = array(Image.open(public_data_path+'/empire.jpg'))
image2 = np.zeros(im.shape)
for i in range(3):
    image2[:,:,i] = filters.gaussian_filter(im[:,:,i],5)

image2 = uint8(image2)
# %%
image2 = array(image2, 'uint8')
# %%
plt.imshow(image2)
# %%

from PIL import Image
from numpy import * 
from scipy.ndimage import filters

im = array(Image.open(public_data_path+'/empire.jpg').convert('L'))
imx = np.zeros(im.shape)
# %%
filters.sobel(im,1,imx)
# %%
imy = zeros(im.shape)
filters.sobel(im,0,imy)
# %%
magnitude = sqrt(imx**2+imy**2)
# %%
