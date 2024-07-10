# Purpose: Create a video from the events of a csv file.

# WARNING: DO NOT USE THIS VERSION ANYMORE! 
# In this version (v1) the framelengths are slightly wrong!

import csv
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


############################ MAIN ##################################
if __name__ == "__main__":
    DATETIME_STR = getDateTimeStr()

    # If a file with the same name already exists
    OVERWRITE_EXISITNG_VIDEOS = False
    print(f"OVERWRITE_EXISITNG_VIDEOS: {OVERWRITE_EXISITNG_VIDEOS}")

    WIDTH = 1280
    HEIGHT = 720
    FPS = 60

    ADD_TIME_TO_FILENAME = True
    VIDEO_FILE_PREFIX = "_v1"

    # Steps of the timestamp: for mikroseconds: 1000000, for milliseconds: 1000
    TIMESTEPS_PER_SECOND = 1_000_000
    # If timestamp in mikroseconds: -> mikroseconds per frame
    TIMESTEPS_PER_FRAME = (1 / FPS) * TIMESTEPS_PER_SECOND
    print(f"Video format: {WIDTH}x{HEIGHT} {FPS}fps")
    print(f"CSV format:   {WIDTH}x{HEIGHT}, {TIMESTEPS_PER_SECOND} timesteps per second")

    # Must be an integer! e.g. 1 or 10
    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    CSV_DIR = Path("../../datasets/Insektenklassifikation")
    # CSV_DIR = Path("../../datasets/juli_instance_segmented")
    OUTPUT_DIR = Path("output/csv_to_video")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    

    # Check for existing mp4 files; Existing files wont be created again (depending on OVERWRITE_EXISITNG_VIDEOS-Option)
    existing_videos_filenames = [file.name for file in OUTPUT_DIR.iterdir() if (file.is_file() and file.name.endswith(".mp4"))]

    # Find all csv files in CSV_DIR
    # csv_filenames = [file.name for file in CSV_DIR.iterdir() if (file.is_file() and file.name.endswith(".csv"))]
    # print(f"Found {len(csv_filenames)} csv files")

    # ... or use only this file:
    csv_filenames = [
        "vieleSchmetterlinge2.csv"
        # "COMBINED1_scale_classify_2022-07-16_19-24-08_teil3.csv"
    ]

    for csv_filename in csv_filenames:
        csv_filestem = csv_filename.replace(".csv", "")
        events_csv_filepath = CSV_DIR / csv_filename
        datetime_filename_part = ("_"+DATETIME_STR) if ADD_TIME_TO_FILENAME else ""
        frametimes_csv_filepath = OUTPUT_DIR / f"{csv_filestem}_{FPS}fps_dvs_frametimes{VIDEO_FILE_PREFIX}{datetime_filename_part}.csv"
        video_filepath = OUTPUT_DIR / f"{csv_filestem}_{FPS}fps_dvs{VIDEO_FILE_PREFIX}{datetime_filename_part}.mp4"

        # Skip existing files
        if not OVERWRITE_EXISITNG_VIDEOS and video_filepath.name in existing_videos_filenames:
            print(f"Skipping existing file: {events_csv_filepath}")
            continue

        print(f"Processing file: {events_csv_filepath}")

        video = cv2.VideoWriter(str(video_filepath), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))
        
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame_start = 0
        video_start = frame_start
        frame_index = 0
        events_processed = 0

        # Find total row count for tracking progress
        row_count = sum(1 for _ in open(events_csv_filepath))
        last_progress_percent = 0

        with open(events_csv_filepath, 'r') as events_csv_file, open(frametimes_csv_filepath, 'w', newline='') as frametimes_csv_file:
            # read events
            reader = csv.reader(events_csv_file)
            # write timestamps of frames
            writer = csv.writer(frametimes_csv_file)
            writer.writerow(("frame_index", "timestamp"))

            # skip header
            next(reader)

            print("Progress (%): ", end="")

            for row_index, row in enumerate(reader):
                # row: (x, y, t, p)
                timestamp = float(row[2])
                if timestamp >= frame_start + TIMESTEPS_PER_FRAME:
                    # -> this event is part of next frame
                    # write current frame to video
                    video.write(frame)
                    # start new frame
                    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    # save frame_start of current frame
                    writer.writerow((frame_index, frame_start))
                    frame_index += 1
                    frame_start = timestamp
                # draw event
                frame[int(row[1]), int(row[0])] = 255

                # print progress
                progress_percent = int(row_index / row_count * 100)
                if progress_percent >= last_progress_percent + PRINT_PROGRESS_EVERY_N_PERCENT:
                    last_progress_percent = progress_percent
                    print(progress_percent, ", ", sep="", end="")

        video.release()

        print(f"\nSaved video: {video_filepath}, frames: {frame_index}, events processed: {row_index}, "\
              f"events per frame: {row_index/frame_index}, length: {(timestamp-video_start)/TIMESTEPS_PER_SECOND}s")
        print(f"Saved csv with frametimes: {frametimes_csv_filepath}")








    







                













