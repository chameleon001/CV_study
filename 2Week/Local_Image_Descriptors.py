from numpy import zeros
#%%
from scipy.ndimage import filters

def compute_harris_response(im, sigma=3):
    """
    Compute the Harris corner  detector response function for each pixel in a graylevel image 
    """

    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0,1), imx)

    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1,0), imy)

    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet/Wtr

#%%   

def get_harris_points(harrisim, min_dist =10, threshold=0.1):
    """
    Retrun corners from a Hrarris response image
    min_dist is the minimum number of pixels separating corners and iamge boundary
    """

    