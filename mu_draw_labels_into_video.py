import cv2
import csv
from datetime import datetime

def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def getBeeName(index):
    bee_names = [
        "Buzzington",
        "Honeydew",
        "Buzzy McBee",
        "Pollenator",
        "Zippy Wing",
        "Nectarina",
        "Fuzzy Buzz",
        "Winged Worker",
        "Buzzberry",
        "Honeywing"
    ]
    if index < len(bee_names):
        return bee_names[index]
    else:
        return "unregistered_bee_" + str(index)


############################ MAIN ##################################
if __name__ == "__main__":

    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    # filename = "1_l-l-l"
    filenames = [
        # "1_l-l-l",
        # "2_l-h-l",
        # "3_m-h-h",
        # "4_m-m-h",
        # "5_h-l-h",
        # "6_h-h-h_filtered",
        "vieleSchmetterlinge2"
    ]

    for filename in filenames:
        print("Processing video", filename, "...")

        # labels_filepath = f"output/{filename}_annotation_instances.csv"
        labels_filepath = f"output/mu_frame_labels_with_ids/{filename}_60fps_dvs_annotation_header.txt"
        # input_video_path = f"../../datasets/muenster_dataset/wacv2024_ictrap_dataset/{filename}_dvs.mp4"
        input_video_path = f"output/csv_to_video/{filename}_60fps_dvs.mp4"
        output_video_path = f"output/videos_with_bbs/{filename}_dvs_bb_instances.mp4"

        # For text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        color = (0, 255, 0)  # BGR color (here, green)

        

        with open(labels_filepath, 'r') as cvs_file:
            # Read bboxes
            reader = csv.DictReader(cvs_file)

            # Open the input video file
            cap = cv2.VideoCapture(input_video_path)

            # Check if the video opened successfully
            if not cap.isOpened():
                raise IOError("Error: Could not open video.")

            # Get video properties
            input_fps = int(cap.get(cv2.CAP_PROP_FPS))
            input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Offset and scale of label coordinates (should usually be offset=0 and scale=1).
            # For drawing BBs on the RGB video
            offset_x = 0
            offset_y = 0
            scale_x = 1
            scale_y = 1

            # Define the output video codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, input_fps, (input_width, input_height))

            annotation_row = next(reader, None) # return None if eof

            frame_index = 0

            # Iterate frames
            while cap.isOpened():
                ret, frame = cap.read()  # Read a frame
                if not ret:
                    break  # Break the loop if no frame is read (end of video)

                # DEBUG: Only render selected timeframe
                start_time_s = 0*60 + 0
                end_time_s = 9999*60 + 0
                frame_time_s = frame_index/input_fps
                if frame_time_s < start_time_s:
                    frame_index += 1
                    continue
                if frame_time_s > end_time_s:
                    break

                while annotation_row is not None:
                    label_frame_index = int(annotation_row["frame_index"])
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
                        x1 = int(offset_x + float(annotation_row["left"]))
                        y1 = int(offset_y + float(annotation_row["top"]))
                        x2 = int(x1 + float(annotation_row["width"]) * scale_x)
                        y2 = int(y1 + float(annotation_row["height"]) * scale_y)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # draw center
                        # cv2.rectangle(frame, (int(float(annotation_row["center_x"])), int(float(annotation_row["center_y"]))), \
                        #             (int(float(annotation_row["center_x"]))+1, int(float(annotation_row["center_y"]))+1), color, 3)

                        # write instance id
                        text = "#" + str(annotation_row["instance_id"])
                        # text = getBeeName(int(annotation_row["instance_id"]))
                        position = (x1, y2+25)  # (x, y) coordinates
                        cv2.putText(frame, text, position, font, font_scale, color, thickness=2)
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