#%%

from scipy import ndimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import Homographies
im = np.array(Image.open('../data/empire.jpg').convert('L'))
H = np.array([[1.4, 0.05, -100], [0.05, 1.5, -100], [0, 0, 1]])
im2 = ndimage.affine_transform(im, H[:2, :2], (H[0,2], H[1,2]))

plt.figure()
plt.gray()
plt.imshow(im2)
plt.show()
# %%
def image_in_image(im1, im2, tp):
    m,n = im1.shpae[:2]
    fp = np.array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    H = Homographies.Haffine_from_points(tp,fp)
    iml_t = ndimage.affine_transform(im1, H[:2,:2], (H[0,2],H[1,2]), im2.shape[:2])
    alpha = (iml_t >0)

    return (1-alpha)*im2 + alpha*iml_t
# %%

im1 = np.array(Image.open('../data/beatles.jpg').convert('L'))
im2 = np.array(Image.open('../data/billboard_for_rent.jpg').convert('L'))

tp = np.array([[264, 548, 540, 264], [40, 36, 605, 605], [1,1,1,1]])

im3 = image_in_image(im1, im2, tp)

plt.figure()
plt.gray()
plt.imshow(im3)
plt.axis('equal')
plt.axis('off')
plt.show()
# %%

m, n = im1.shape[:2]
fp = np.array([[0,m,m,0], [0,0,n,n], [1,1,1,1]])

tp2 = tp[:,:3]
fp2 = fp[:,:3]

H = Homographies.Haffine_from_points(tp2,fp2)
