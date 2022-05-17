from PIL import Image
import numpy as np


def enlarge_img(path, target_size, background):
    img = Image.open(path)
    img_np = np.asarray(img)
    h, w, c= img_np.shape

    # th = 1080
    # tw = 1920
    th, tw = target_size
    if background == "black" or "b" or "B":
        out = np.zeros([th, tw, c], dtype=np.uint8)
    if background == "white" or "w" or "W":
        out = np.ones([th, tw, c], dtype=np.uint8)
        out = out * 255

    left = int(tw/2-w/2)
    right = int(tw/2+w/2)
    up = int(th/2-h/2)
    bottom = int(th/2+h/2)

    if c == 4:  # RGBD or RGBA
        out[up:bottom, left:right, 0] = img_np[:, :, 0]
        out[up:bottom, left:right, 1] = img_np[:, :, 1]
        out[up:bottom, left:right, 2] = img_np[:, :, 2]
        out[up:bottom, left:right, 3] = img_np[:, :, 3]
    if c == 3:  # RGB
        out[up:bottom, left:right, 0] = img_np[:, :, 0]
        out[up:bottom, left:right, 1] = img_np[:, :, 1]
        out[up:bottom, left:right, 2] = img_np[:, :, 2]
    if c == 1:  # gray image
        out[up:bottom, left:right, 0] = img_np[:, :, 0]
    
    return out

if __name__ == "__main__":
    path = r'E:\Pycharm\project001\6D_Pose_Annotator\our_data\calibration\group10\kinect_ir_colorized\1.png'
    out = enlarge_img(path, [1080, 1920], 'w')

    out = Image.fromarray(out)
    out.save('C:\\Users\\atlas\\Desktop\\tse.png')