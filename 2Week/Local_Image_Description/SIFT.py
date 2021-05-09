#%%
from ... import my_Util
# %%
## SIFT

def process_iamge(image_name, result_name, params= '--edge-thresh10--peak-thresh5'):
    # process an image and save the result in a file

    if image_name[-3:] != 'pgm':
        #create a pgm file
        im = Image.open(image_name).convert('L')
        im.save('tmp.pgm')
        image_name = 'tmp.pgm'

    cmmd = str()