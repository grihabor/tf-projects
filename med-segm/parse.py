from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os.path
import skimage.morphology as morph

seg_filename = '../data/manual_seg/manual_seg_32points_pat'
img_filename = '../data/mrimages/sol_yxzt_pat'


def fill_poly(poly_y, poly_x, shape):
    mask = np.zeros(shape)
    mask[poly_y.astype(np.int), poly_x.astype(np.int)] = 1
    mask = morph.convex_hull_image(mask).astype(np.float)
    return mask



count = 0
i = 1
while True:
    seg = seg_filename + str(i) + '.mat'
    img = img_filename + str(i) + '.mat'
    i += 1
 
    print(seg)
    if not os.path.isfile(seg):
        break


    slices = loadmat(img)['sol_yxzt']
    segmentations = loadmat(seg)['manual_seg_32points']
    
    for z in range(slices.shape[2]):
        for t in range(slices.shape[3]):
            
            slice = slices[:, :, z, t]
            segmentation = segmentations[z, t]

            if segmentation.shape[0] > 1:
                segm = np.zeros((2, 33, 2))
                segm[0] = segmentation[:33, :]
                segm[0, 32] = segm[0, 0]

                segm[1] = segmentation[32:, :]
                segm[1, 0] = segm[1, -1]
                
                plt.subplot(131)
                plt.plot(segm[0, :, 0], segm[0, :, 1])
                plt.plot(segm[1, :, 0], segm[1, :, 1])
                plt.imshow(slice, cmap='gray')
                
                plt.subplot(132)
                mask = fill_poly(segm[0, :, 1], segm[0, :, 0], slice.shape[:2])
                plt.imshow(mask, cmap='gray', interpolation='None')

                plt.subplot(133)
                mask = fill_poly(segm[1, :, 1], segm[1, :, 0], slice.shape[:2])
                plt.imshow(mask, cmap='gray', interpolation='None')

                

                mngr = plt.get_current_fig_manager()
                # to put it into the upper left corner for example:
                mngr.window.setGeometry(100,100, 1000, 800)

                plt.show()
                count += 1

print('count:', count)



