
# coding: utf-8

# # Milestone Report Part 1.3: Training data processing
# 
# This file is get the data ready for training the CNN algorithm.
# 
# Due to the large amount of data (400GB in total, yeah I know what you are thinking, it is not that large but that is half of my disk already), all input are stored outside of repository folder by name "cap2input". I will have a independent document on how to re-run this on your own computer, with all the data storage explained...
# 
# Note that I add a 300 limit to the iteration, meaning that only 300 out of 1400 data samples are selected for the training. The reason for this is running through all 1400 data will take up to 5 days to finish. I simply do not have that much time at this moment. However, in the future I will test this on the whole dataset as well as refining some pre-processing procedure.
# 

# In[1]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi

get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../../cap2input"]).decode("utf8"))


# In[2]:

def read_ct_scan(folder_name):
        # Read the slices from the dicom file
        slices = [dicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]
        
        # Sort the dicom slices in their respective order
        slices.sort(key=lambda x: int(x.InstanceNumber))
        
        # Get the pixel values for all the slices
        image = np.stack([s.pixel_array for s in slices])
        intercept = int(slices[0].RescaleIntercept)
        image[image <= -1900] = -1000-intercept
        
        for slice_number in range(len(slices)):
        
            intercept = float(slices[slice_number].RescaleIntercept)
            slope = int(slices[slice_number].RescaleSlope)
            
            #print(intercept,slope)
        
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                
            image[slice_number] = image[slice_number].astype(np.int16)
            intercept = np.int16(intercept)

            image[slice_number] += intercept.astype(image[slice_number].dtype)
        print image.dtype
            
        return np.array(image, dtype=np.int16)
    
THRES = -400

def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < THRES
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(12)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = -1000
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone) 
        
    return im

def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(single_slice) for single_slice in ct_scan])

def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)
        
def plot_3d(image, threshold=-1300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces,_,_ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


# In[3]:

from skimage.morphology import opening
import cv2
import math

IMG_SIZE_PX = 50
SLICE_COUNT = 20

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def mean(a):
    return sum(a) / len(a)

def process_data(patient, labels_df, img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT, vis=False):
    path = data_dir + patient + '/'
    print path
    ct_scan = read_ct_scan(path)
    
    segmented_ct_scan = segment_lung_from_ct_scan(ct_scan)
    segmented_ct_scan[segmented_ct_scan < THRES] = -1000
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   
#    selem = ball(3) #was 2
#    segmented_ct_scan = opening(segmented_ct_scan, selem)
    
    if vis == True:
        plot_3d(segmented_ct_scan, THRES)
        print(segmented_ct_scan[0].shape, len(segmented_ct_scan))
        
    slices_b = segmented_ct_scan
    new_slices = []
    slices_b = [cv2.resize(np.array(each_slice),(img_px_size,img_px_size)) for each_slice in slices_b]
    
    chunk_sizes = int(math.ceil(len(slices_b) / hm_slices))
    for slice_chunk in chunks(slices_b, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)
    
    while (len(new_slices) < hm_slices):
        new_slices.append(new_slices[-1])
        
    while (len(new_slices) > hm_slices):
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val
    
    label_b = labels_df.get_value(patient, 'cancer')

    if label_b == 1: label_b=np.array([0,1])
    elif label_b == 0: label_b=np.array([1,0])
        
    if vis == True:
        print(patient, label_b)
    print(len(slices_b), len(new_slices))
        
    return np.array(new_slices),label_b


data_dir = '../../cap2input/stage1/'
patients = os.listdir(data_dir)
#patient = 'ea7373271a2441b5864df2053c0f5c3e'
labels = pd.read_csv('../../cap2input/stage1_labels.csv', index_col=0)

much_data = []
limit = 300
for num,patient in enumerate(patients):
    if num > limit:
        break
    if num % 100 == 0:
        print(num)
    try:
        img_data,label_c = process_data(patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
        #print(img_data.shape,label)
        much_data.append([img_data,label_c,patient])
    except KeyError as e:
        print('This is unlabeled data!')

np.save('../../cap2input/output/muchdata-{}-{}-{}-b.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), much_data)

#process_data(patient, labels, vis = True)



# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



