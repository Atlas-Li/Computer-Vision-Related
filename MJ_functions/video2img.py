import cv2
import math
import os
import numpy as np
import tqdm

def video2imgs(video_in, out_root, just_check_video_info=True):
    video = cv2.VideoCapture(video_in)
    is_open = video.isOpened()

    if is_open:
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        total_sec = math.ceil(num_frames/fps)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video: ", video_in)
        print("{:<15}: {}\n{:<15}: {}\n{:<15}: {}\n{:<15}: {}\n{:<15}: {}"
              .format("FPS", fps, "Total frames", num_frames,"Total seconds", total_sec,
                      "Width", width, "Height", height))

    
    if just_check_video_info == False:
        sec_idx = 1
        frame_idx = 1
        # i = 1  # start frame
        while(is_open):
            flag, frame = video.read()
            out_name = out_root + '\\sec_' + str(sec_idx).rjust(2,'0') + '_frame_' + str(frame_idx).rjust(2,'0') + '.png'
            # print(out_name)
            if flag == True:
                cv2.imwrite(out_name, frame)
                if frame_idx < fps:
                    frame_idx += 1
                elif frame_idx == fps:
                    sec_idx += 1
                    frame_idx = 1
                    
            else:
                break
        
    video.release()
    

def crop_imgs(root_in, root_out, xmin, xmax, ymin, ymax):
    images = os.listdir(root_in)
    images = sorted(images)
    for img in tqdm.tqdm(images):
        cur_path = os.path.join(root_in, img)
        image = cv2.imread(cur_path)
        image = image[xmin:xmax, ymin:ymax, :]
        # print("New size:", image.shape)
        
        out_path = os.path.join(root_out, "crop_"+img)
        cv2.imwrite(out_path, image)
    
def imgs2video(root_in, out_path, suffix, fps, size):
    video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    for item in sorted(os.listdir(root_in)): 
        if item.endswith('.'+suffix):
            item = os.path.join(root_in, str(item))
            img = cv2.imread(item)
            video.write(img)
    video.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    video_in = r"C:\Users\atlas\Desktop\grasp\video1414578094.mp4"
    out_root = r"C:\Users\atlas\Desktop\grasp\v2i"
    crop_out_root = r"C:\Users\atlas\Desktop\grasp\crop_v2i"
    crop_video = r"C:\Users\atlas\Desktop\grasp\crop.avi"
    
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    if not os.path.exists(crop_out_root):
        os.makedirs(crop_out_root)
    
    # video2imgs(video_in, out_root, just_check_video_info=False)
    # crop_imgs(out_root, crop_out_root, 510, 690, 850, 1100)
    imgs2video(crop_out_root, crop_video, "png", 25, (250, 180))
    
    
    
    