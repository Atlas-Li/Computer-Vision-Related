import cv2

def video2imgs(video_in, out_root, just_check_video_info=True):
    video = cv2.VideoCapture(video_in)
    is_open = video.isOpened()

    if is_open:
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video: ", video_in)
        print("FPS: {}\nTotal frames: {}\nWidth: {}\nHeight: {}"
              .format(fps,num_frames,width,height))
    
    if just_check_video_info == False:
        i = 1  # start frame
        while(is_open):
            flag, frame = video.read()
            out_name = out_root + '\\' + str(i).rjust(5,'0') + '.jpg'
            # print(out_name)
            if flag == True:
                cv2.imwrite(out_name, frame)
                i += 1
            else:
                break

    video.release()
    