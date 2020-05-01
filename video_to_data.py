import argparse
import cv2
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video_dir', dest='video_dir', type=str,
    help='Directory of videos to generate data from')
parser.add_argument('-o', '--output_dir', dest='output_dir', type=str,
    help='Directory to output frames to')
parser.add_argument('-s', '--frame_skip', dest='frame_skip', type=int,
    default=int(24*2.5), help='How many frames to skip in between captures')
parser.add_argument('-b', '--start_time', dest='start_time', type=float,
    default=0, help='How many seconds into each video to start')
parser.add_argument('-e', '--end_time', dest='end_time', type=float,
    default=0, help='How many seconds into each video to end')

VIDEO_TYPES = ('*.mp4', '*.mkv')

def video_to_frames(video_path, output_dir, frame_skip, start_time=0, end_time=0):
    video_name = os.path.basename(video_path)
    video_name = video_name[:video_name.rfind('.')]
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps)
    if end_time == 0:
        end_frame = n_frames
    elif end_time > 0:
        end_frame = int(end_time * fps)
    else:
        end_frame = int(n_frames + end_time * fps)

    success = True
    frame_idx = 0
    used_frame_idx = 0
    
    for _ in range(start_frame):
        success, _ = cap.read()
        frame_idx += 1
        if not success:
            break
    
    while success: 
        if frame_idx >= end_frame:
            break
        
        success, frame = cap.read()
  
        if frame_idx % frame_skip == 0:
            cv2.imwrite(os.path.join(output_dir,
                f'{video_name}_frame_{used_frame_idx}.png'), frame)
            used_frame_idx += 1
  
        frame_idx += 1
    
    return used_frame_idx

if __name__ == '__main__':
    args = parser.parse_args()

    video_paths = []
    for file_type in VIDEO_TYPES:
        video_files = glob.glob(os.path.join(args.video_dir, file_type))
        video_paths.extend(video_files)
    
    for i, video_path in enumerate(video_paths):
        print(f'Working on video #{i+1}...')
        n_frames = video_to_frames(video_path, args.output_dir, args.frame_skip,
                                   args.start_time, args.end_time)
        print(f'{n_frames} frames written')
    print('Done')