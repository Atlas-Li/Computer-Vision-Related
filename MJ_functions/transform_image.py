import numpy as np
from PIL import Image
import logging
import argparse

'''
These codes are pre-process for calibration
    if 2 images are not having the same size, we cannot feed 
them into MATLAB, we need to make them to have the same size first

- compute_trans_matrix:
    based on the method (1,2,3), compute the transformation matrix
- trans_img:
    generate a new image that has the same size

@author:    Mingjun Li
email:      mingli@clarkson.edu
'''

''' set logger '''
logger = logging.getLogger('Transform Image')
fmt = "[%(name)s]%(levelname)s Line%(lineno)d: %(message)s"
logging.basicConfig(level=logging.INFO, format=fmt)


def compute_trans_matrix(
    large_image_path, 
    small_image_path, 
    method
):
    '''
    method 1 -> scale up small image, shift in x_direction, zero-padding
    method 2 -> scale down large image, shift in y_direction, zero-padding
    method 3 -> method 1 without scaling
    '''

    assert method in [1,2,3], "Need a method for transformation method"
    
    img1 = Image.open(large_image_path)
    img2 = Image.open(small_image_path)

    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)

    if len(img1_np.shape) == 3:
        h1, w1, c1 = img1_np.shape
    else:
        h1, w1 = img1_np.shape

    if len(img2_np.shape) == 3:
        h2, w2, c2 = img2_np.shape
    else:
        h2, w2 = img2_np.shape

    logger.info("Load {}".format(large_image_path))
    logger.info("large image shape {}".format(img1_np.shape))
    logger.info("Load {}".format(large_image_path))
    logger.info("small image shape {}".format(img2_np.shape))

    if method == 1:
        logger.info("Method 1")
        scale_factor = h1/h2
        shift_factor = (w1-w2*scale_factor)/2
        trans_matrix = np.array([[scale_factor, 0, shift_factor],
                                [0, scale_factor, 0],
                                [0, 0, 1]])
    elif method == 2:
        logger.info("Method 2")
        scale_factor = w2/w1
        shift_factor = (h2-h1*scale_factor)/2
        trans_matrix = np.array([[scale_factor, 0, 0],
                                [0, scale_factor, shift_factor],
                                [0, 0, 1]])
    else:
        logger.info("Method 3")
        shift_factor_x = (w1-w2)/2
        shift_factor_y = (h1-h2)/2
        trans_matrix = np.array([[1, 0, shift_factor_x],
                                [0, 1, shift_factor_y],
                                [0, 0, 1]])

    return img1_np, img2_np, trans_matrix


def trans_img(img_np, M, out_h, out_w):

    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        if c > 3:
            c = 3
    else:
        h, w = img_np.shape

    # corners in old img
    c1 = np.array([[0],[0],[1]])        # top left
    c2 = np.array([[w-1],[0],[1]])      # top right
    c3 = np.array([[0],[h-1],[1]])      # bottom left
    c4 = np.array([[w-1],[h-1],[1]])    # top left

    corner = np.hstack((c1, c2, c3, c4))

    # get corner in new img in homo
    corner_new = np.dot(M, corner)

    # get true location of 4 new corners
    x_locations = (corner_new[0,:]/corner_new[2,:]).astype(np.int16)
    y_locations = (corner_new[1,:]/corner_new[2,:]).astype(np.int16)
    c1_new = np.array([[min(0,min(x_locations))],[min(0,min(y_locations))]])
    c2_new = np.array([[max(x_locations)],[min(0,min(y_locations))]])
    c3_new = np.array([[min(0,min(x_locations))],[max(y_locations)]])
    c4_new = np.array([[max(x_locations)],[max(y_locations)]])

    xmin, xmax = c1_new[0,0], c2_new[0,0]
    ymin, ymax = c1_new[1,0], c3_new[1,0]

    # domain of the new image
    y, x = np.meshgrid(np.arange(ymax-ymin+1), np.arange(xmax-xmin+1), indexing='ij')
    new_shape = y.shape

    x, y = x.flatten(), y.flatten()

    # pixel locations of the new image [x, y, 1].T
    img_new = np.vstack((x, y, np.ones(x.shape[0])))

    # compute the pixel values from old image
    old_img_hat = np.dot(np.linalg.inv(M), img_new) # 3 * N
    onlyX = old_img_hat[0,:]/old_img_hat[2,:]
    onlyY = old_img_hat[1,:]/old_img_hat[2,:]

    X_matrix = np.reshape(onlyX, new_shape)
    Y_matrix = np.reshape(onlyY, new_shape)

    if c == 3:
        out = np.zeros([new_shape[0], new_shape[1], c])
        # interp2D
        for row in range(out.shape[0]):
            for col in range(out.shape[1]):
                x_in_old = int(X_matrix[row, col])
                y_in_old = int(Y_matrix[row, col])
                if x_in_old < 0 or y_in_old < 0 :
                # if x_in_old < 0 or y_in_old < 0 or x_in_old >=h or y_in_old >= w:
                    continue
                out[row, col,:] = img_np[y_in_old, x_in_old, 0:c]

        result = np.zeros([out_h, out_w, c]).astype(np.uint8)
        result[0:out.shape[0], 0:out.shape[1], :] = out[:, :, :]
    else:
        out = np.zeros([new_shape[0], new_shape[1]])
        # interp2D
        for row in range(out.shape[0]):
            for col in range(out.shape[1]):
                x_in_old = int(X_matrix[row, col])
                y_in_old = int(Y_matrix[row, col])
                if x_in_old < 0 or y_in_old < 0 :
                # if x_in_old < 0 or y_in_old < 0 or x_in_old >=h or y_in_old >= w:
                    continue
                out[row, col,:] = img_np[y_in_old, x_in_old]

        result = np.zeros([out_h, out_w]).astype(np.uint8)
        result[0:out.shape[0], 0:out.shape[1]] = out[:, :]

    return result



if __name__ == "__main__":

    large_path = r'E:\Pycharm\project001\6D_Pose_Annotator\our_data\calibration\group10\kinect_color\6.png'
    # large_path = r'C:\Users\atlas\Desktop\grasp\1000\1000-shell0-c0-R\kinect_color\kinect_color_arc0_image7.png'
    # small_path = r'E:\Pycharm\project001\6D_Pose_Annotator\our_data\calibration\group10\kinect_ir_colorized\6.png'
    small_path = r'C:\Users\atlas\Pictures\Saved Pictures\009.jpg'

    img1_np, img2_np, trans_matrix = compute_trans_matrix(large_path, small_path, 3)

    img_new = trans_img(img2_np, trans_matrix, img1_np.shape[0], img1_np.shape[1]) # for method 1,3
    # img_new = trans_img(img1_np, trans_matrix, img2_np.shape[0], img2_np.shape[1])   # for method 2
    Image.fromarray(img_new).show()


