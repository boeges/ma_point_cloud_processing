# Purpose: Draw bboxes into an existing video from an existing annotation file.
# Used for verification purposes and to assign classes to instances.
# Position of bboxes can be shifted and scaled. Useful when applying bboxes from a dvs video to a parallel captured rgb video.

import cv2
import csv
from datetime import datetime
from pathlib import Path


def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


############################ MAIN ##################################
if __name__ == "__main__":

    WIDTH = 1280
    HEIGHT = 720
    PRINT_PROGRESS_EVERY_N_PERCENT = 1


    ############### PF #############
    # FPS = 60
    # DRAW_VIEWBOX = False

    # # Format from DarkLabel software: (frame_index, classname, instance_id, is_difficult, x, y, w, h) 
    # #                                      0            1          2            3         4  5  6  7
    # LABELS_CSV_HAS_HEADER = False
    # LABELS_CSV_FRAME_COL = 0 # frame index
    # LABELS_CSV_CLASS_COL = 1 # class name
    # LABELS_CSV_ID_COL = 2 # instance id
    # LABELS_CSV_IS_CONFIDENT_COL = 3
    # LABELS_CSV_X_COL = 4
    # LABELS_CSV_Y_COL = 5
    # LABELS_CSV_W_COL = 6
    # LABELS_CSV_H_COL = 7
    # BB_IS_CONFIDENT_WHEN_MATCHES = "0" # is_difficult: 0(=confident)/1(=difficult)

    # # Paths
    # LABELS_CSV_DIR = Path("output/video_annotations")
    # # LABELS_CSV_FILENAME = "{filestem}_60fps_dvs_sep.txt"
    # LABELS_CSV_FILENAME = "{filestem}_60fps_dvs_v3_sep.txt"

    # INPUT_VIDEO_DIR = Path("output/csv_to_video")
    # # INPUT_VIDEO_FILENAME = "{filestem}_60fps_dvs.mp4"
    # INPUT_VIDEO_FILENAME = "{filestem}_60fps_dvs_v3.mp4"

    # OUTPUT_VIDEO_DIR = Path("output/videos_with_bbs")
    # OUTPUT_VIDEO_FILENAME = "{filestem}_dvs_bb_instances.mp4"

    ##################################

    ############### MU #############
    FPS = 100
    DRAW_VIEWBOX = True

    # Format from DarkLabel software: (frame_index, classname, instance_id, is_difficult, x, y, w, h) 
    #                                      0            1          2            3         4  5  6  7
    LABELS_CSV_HAS_HEADER = True
    LABELS_CSV_FRAME_COL = 0 # frame index
    LABELS_CSV_CLASS_COL = 1 # class name
    LABELS_CSV_ID_COL = 2 # instance id
    LABELS_CSV_IS_CONFIDENT_COL = 3
    LABELS_CSV_X_COL = 4
    LABELS_CSV_Y_COL = 5
    LABELS_CSV_W_COL = 6
    LABELS_CSV_H_COL = 7
    BB_IS_CONFIDENT_WHEN_MATCHES = "0" # is_difficult: 0(=confident)/1(=difficult)

    # Default values
    # OFFSET_X = 0
    # OFFSET_Y = 0
    # SCALE_X = 1
    # SCALE_Y = 1

    # Offset and scale of label coordinates (should usually be offset=0 and scale=1).
    # For drawing BBs on the RGB video
    OFFSET_X = -50
    OFFSET_Y = 100
    SCALE_X = 1.35
    SCALE_Y = 1.35
    BB_PADDING = 20 # px in each direction

    # Paths
    LABELS_CSV_DIR = Path("output/mu_frame_labels_with_ids")
    LABELS_CSV_FILENAME = "{filestem}_annotation_instances_classes.csv"

    INPUT_VIDEO_DIR = Path("../../datasets/muenster_dataset/wacv2024_ictrap_dataset")
    # INPUT_VIDEO_FILENAME = "{filestem}_dvs.mp4"
    INPUT_VIDEO_FILENAME = "{filestem}_rgb.mp4"

    OUTPUT_VIDEO_DIR = Path("output/videos_with_bbs")
    # OUTPUT_VIDEO_FILENAME = "{filestem}_dvs_bb_instances.mp4"
    OUTPUT_VIDEO_FILENAME = "{filestem}_rgb_bb_instances.mp4"

    ##################################

    filestems = [
        # "1_l-l-l",
        # "2_l-h-l",
        "3_m-h-h",
        # "4_m-m-h",
        # "5_h-l-h",
        # "6_h-h-h_filtered",
        # "hauptsächlichBienen1"
        # "libellen1"
        # "vieleSchmetterlinge2"
    ]

    for filestem in filestems:
        print("Processing video", filestem, "...")

        labels_filepath = LABELS_CSV_DIR / LABELS_CSV_FILENAME.format(filestem=filestem)
        input_video_path = INPUT_VIDEO_DIR / INPUT_VIDEO_FILENAME.format(filestem=filestem)
        output_video_path = OUTPUT_VIDEO_DIR / OUTPUT_VIDEO_FILENAME.format(filestem=filestem)

        # For text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        color = (0, 255, 0)  # BGR color (here, green)


        with open(labels_filepath, 'r') as cvs_file:
            # Read bboxes
            reader = csv.reader(cvs_file)

            if LABELS_CSV_HAS_HEADER:
                next(reader)

            # Open the input video file
            cap = cv2.VideoCapture(str(input_video_path))


            # Check if the video opened successfully
            if not cap.isOpened():
                raise IOError("Error: Could not open video.")

            # Get video properties
            input_fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define the output video codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, input_fps, (input_width, input_height))

            annotation_row = next(reader, None) # return None if eof

            frame_index = 0
            last_progress_percent = -1

            print("Progress (%): ", sep="", end="")

            # Iterate frames
            while cap.isOpened():
                ret, frame = cap.read()  # Read a frame
                if not ret:
                    break  # Break the loop if no frame is read (end of video)

                # Print progress
                progress_percent = int(frame_index / frame_count * 100)
                if progress_percent >= last_progress_percent + 1:
                    last_progress_percent = progress_percent
                    print(progress_percent, ", ", sep="", end="")

                # DEBUG: Only render selected timeframe
                # start_time_s = 0*60 + 50
                # end_time_s = 0*60 + 58
                start_time_s = 0
                end_time_s = 999999999
                frame_time_s = frame_index/input_fps
                if frame_time_s < start_time_s:
                    frame_index += 1
                    continue
                if frame_time_s > end_time_s:
                    break

                while annotation_row is not None:
                    label_frame_index = int(annotation_row[LABELS_CSV_FRAME_COL])
                    if label_frame_index > frame_index:
                        # If the label frame index is for a future video frame
                        break
                    if label_frame_index == frame_index:
                        # If label frame index matches video frame index: Draw bb and other info
                        color = (0, 255, 0)
                        # if annotation_row["confidence"] == "certain":
                        #     color = (0, 255, 0)
                        # else:
                        #     color = (0, 127, 255)

                        # Draw bb
                        x1 = int(OFFSET_X + float(annotation_row[LABELS_CSV_X_COL]) * SCALE_X - BB_PADDING) 
                        y1 = int(OFFSET_Y + float(annotation_row[LABELS_CSV_Y_COL]) * SCALE_Y - BB_PADDING)
                        x2 = int((x1+BB_PADDING) + float(annotation_row[LABELS_CSV_W_COL]) * SCALE_X + BB_PADDING)
                        y2 = int((y1+BB_PADDING) + float(annotation_row[LABELS_CSV_H_COL]) * SCALE_Y + BB_PADDING)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # draw center
                        # cv2.rectangle(frame, (int(float(annotation_row["center_x"])), int(float(annotation_row["center_y"]))), \
                        #             (int(float(annotation_row["center_x"]))+1, int(float(annotation_row["center_y"]))+1), color, 3)

                        # write instance id
                        text = f"#{annotation_row[LABELS_CSV_ID_COL]: >3}, {annotation_row[LABELS_CSV_CLASS_COL]}"
                        # text = getBeeName(int(annotation_row["instance_id"]))
                        position = (x1, y2+25)  # (x, y) coordinates
                        cv2.putText(frame, text, position, font, font_scale, color, thickness=2)

                        if DRAW_VIEWBOX:
                            x1 = int(OFFSET_X)
                            y1 = int(OFFSET_Y)
                            x2 = int(x1 + WIDTH * SCALE_X)
                            y2 = int(y1 + HEIGHT * SCALE_Y)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    else:
                        # label_frame_index < frame_index
                        # Iterate until matching frame has been found
                        pass
                    annotation_row = next(reader, None)
                
                # Write the processed frame to the output video
                out.write(frame)
                frame_index += 1

                if frame_index/input_fps > 99999:
                    # stop after x seconds
                    break

            # Release the video capture and writer objects
            cap.release()
            out.release()

            print(f"Created video {output_video_path}!")