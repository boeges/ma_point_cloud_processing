# Funktioniert nicht!

import csv
import math
from datetime import datetime

def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


############################ MAIN ##################################
if __name__ == "__main__":

    PRINT_PROGRESS_EVERY_N_PERCENT = 1
    CSV_HAS_HEADER_ROW = False

    # TIME_SCALE = (42*60+43)/(42*60+30)
    TIME_SCALE = 1.00
    print("time_scale:", TIME_SCALE)

    # filename = "1_l-l-l"
    filenames = [
        # "1_l-l-l",
        # "2_l-h-l",
        # "3_m-h-h",
        # "4_m-m-h",
        # "5_h-l-h",
        # "6_h-h-h_filtered",
        "hauptsächlichBienen1"
        # "vieleSchmetterlinge2"
    ]

    for filename in filenames:
        print("Processing label file", filename, "...")

        input_labels_filepath = f"output/csv_to_video/{filename}_60fps_dvs_annotation.txt"
        output_labels_filepath = f"output/csv_to_video/{filename}_60fps_dvs_annotation.corrected.txt"

        with open(input_labels_filepath, 'r') as input_labels_file, open(output_labels_filepath, 'w', newline='') as output_labels_file:
            # Read bboxes
            reader = csv.reader(input_labels_file)
            writer = csv.writer(output_labels_file)

            if CSV_HAS_HEADER_ROW:
                rrow = next(reader)
                writer.writerow(rrow)

            rrow = next(reader, None)
            writer.writerow(rrow)

            prev_frame_index = int(rrow[0])
            prev_frame_rrows = []

            rows_read = 0
            rows_written = 0
            gaps_filled = 0

            for rrow in reader:
                frame_index = int(rrow[0])
                scaled_frame_index = math.floor(frame_index * TIME_SCALE)
                
                frame_index = frame_index + gaps_filled
                if rows_read < 10:
                    print(f"{prev_frame_index} {frame_index} {scaled_frame_index}")

                if frame_index > prev_frame_index:
                    # new frame


                    if scaled_frame_index - frame_index >= 1:
                        # there is a gap of ONE frame
                        # fill gap with duplicated bboxes. one for each instance
                        for idrow in prev_frame_rrows:
                            writer.writerow([frame_index ,*idrow[1:]])
                            rows_written += 1

                        gaps_filled += 1
                        print(f"filled row gap at {frame_index}")
                        

                    prev_frame_rrows.clear()

                writer.writerow([frame_index ,*rrow[1:]])
                rows_read += 1
                rows_written += 1
                prev_frame_index = frame_index
                prev_frame_rrows.append(rrow) # add instance row

        print(f"Created label file {output_labels_filepath}! Rows read: {rows_read}, rows written: {rows_written}")


