# Purpose: Create a video from the events of a csv file.

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

    # !!!
    EVENTS_CSV_X_COL = 0
    EVENTS_CSV_Y_COL = 1
    EVENTS_CSV_P_COL = 2
    EVENTS_CSV_T_COL = 3

    ADD_TIME_TO_FILENAME = False
    VIDEO_FILE_PREFIX = "_v3"

    # Steps of the timestamp: for mikroseconds: 1000000, for milliseconds: 1000
    TIMESTEPS_PER_SECOND = 1_000_000
    # If timestamp in mikroseconds: -> mikroseconds per frame
    TIMESTEPS_PER_FRAME = (1 / FPS) * TIMESTEPS_PER_SECOND
    print(f"Video format: {WIDTH}x{HEIGHT} {FPS}fps")
    print(f"CSV format:   {WIDTH}x{HEIGHT}, {TIMESTEPS_PER_SECOND} timesteps per second")

    # Must be an integer! e.g. 1 or 10
    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    # CSV_DIR = Path("../../datasets/Insektenklassifikation")
    CSV_DIR = Path("../../aufnahmen/2024-07-18_bunter_garten/exported_csv")
    # CSV_DIR = Path("../../datasets/juli_instance_segmented")
    OUTPUT_DIR = Path("output/csv_to_video")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # For text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (127, 127, 127)  # BGR color 


    # Check for existing mp4 files; Existing files wont be created again (depending on OVERWRITE_EXISITNG_VIDEOS-Option)
    existing_videos_filenames = [file.name for file in OUTPUT_DIR.iterdir() if (file.is_file() and file.name.endswith(".mp4"))]

    # Find all csv files in CSV_DIR
    csv_filenames = [file.name for file in CSV_DIR.iterdir() if (file.is_file() and file.name.endswith(".csv"))]
    print(f"Found {len(csv_filenames)} csv files")

    # ... or use only this file:
    csv_filenames = [
        # "vieleSchmetterlinge2.csv",
        # "COMBINED1_scale_classify_2022-07-16_19-24-08_teil3.csv",
        # "mb-dra1-1.csv",
        "mb-bum1-1.csv",
        "mb-bum1-2.csv",
        "mb-bum1-3.csv",
        "mb-bum1-4.csv",
        "mb-bum1-5.csv",
        "mb-bum1-6.csv",
        "mb-bum1-7.csv",
        "mb-bum1-8.csv",
        "mb-bum1-9.csv",
        "mb-bum1-10.csv",
        "mb-bum2-1.csv",
        "mb-bum2-2.csv",
        "mb-bum2-3.csv",
        "mb-bum2-4.csv",
    ]

    for csv_filename in csv_filenames:
        csv_filestem = csv_filename.replace(".csv", "")
        csv_filepath = CSV_DIR / csv_filename
        datetime_filename_part = ("_"+DATETIME_STR) if ADD_TIME_TO_FILENAME else ""
        video_filepath = OUTPUT_DIR / f"{csv_filestem}_{FPS}fps_dvs{VIDEO_FILE_PREFIX}{datetime_filename_part}.mp4"

        # Skip existing files
        if not OVERWRITE_EXISITNG_VIDEOS and video_filepath.name in existing_videos_filenames:
            print(f"Skipping existing file: {csv_filepath}")
            continue

        print(f"Processing file: {csv_filepath}")
        
        video = cv2.VideoWriter(str(video_filepath), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        next_frame_start = 0
        video_start = next_frame_start
        frame_index = 0
        events_processed = 0

        # Find total row count for tracking progress
        row_count = sum(1 for _ in open(csv_filepath))
        last_progress_percent = 0

        with open(csv_filepath, 'r') as csv_file:
            reader = csv.reader(csv_file)

            # skip header (if there is no header this will just skip the first event)
            next(reader)

            # get frame start
            row = next(reader)
            next_frame_start = float(row[2]) + TIMESTEPS_PER_FRAME
            print(f"first frame start: {next_frame_start-TIMESTEPS_PER_FRAME}, next frame start: {next_frame_start}")

            print("Progress (%): ", end="")

            for row_index, row in enumerate(reader):
                # row: (x, y, t, p) or (x, y, p, t)
                timestamp = float(row[EVENTS_CSV_T_COL])
                if timestamp >= next_frame_start:
                    # event is part of next frame

                    next_frame_start += TIMESTEPS_PER_FRAME
                    # write current frame to video
                    video.write(frame)
                    # start new frame
                    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    time_as_str = f"{int((frame_index / FPS) // 60):0>3}:{(frame_index // FPS):0>2}"
                    cv2.putText(frame, f"frame:{frame_index: >9}  time:{time_as_str}", (5, 20), font, font_scale, font_color, thickness=1)

                    frame_index += 1


                # draw event
                frame[int(row[EVENTS_CSV_Y_COL]), int(row[EVENTS_CSV_X_COL])] = 255

                # print progress
                progress_percent = int(row_index / row_count * 100)
                if progress_percent >= last_progress_percent + PRINT_PROGRESS_EVERY_N_PERCENT:
                    last_progress_percent = progress_percent
                    print(progress_percent, ", ", sep="", end="")

                # if frame_index / FPS > 5:
                #     break

        video.release()

        print(f"\nSaved file: {video_filepath}, frames: {frame_index}, events processed: {row_index}, "\
              f"events per frame: {row_index/frame_index}, length: {(timestamp-video_start)/TIMESTEPS_PER_SECOND}s")








    







                













