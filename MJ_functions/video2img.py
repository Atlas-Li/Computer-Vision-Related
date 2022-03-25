import cv2
import math

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
    
if __name__ == "__main__":
    video_in = r"C:\Users\atlas\Desktop\test.mp4"
    out_root = r"C:\Users\atlas\Desktop\v2i"
    video2imgs(video_in, out_root, just_check_video_info=False)
    