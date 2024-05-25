# In den Labels von DarkLabel werden IDs manchmal mehrfach vergeben. Um die Flugbahnen zu trennen wird dieses Skript benutzt.

import csv
from datetime import datetime

def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


############################ MAIN ##################################
if __name__ == "__main__":

    PRINT_PROGRESS_EVERY_N_PERCENT = 1
    CSV_HAS_HEADER_ROW = False
    MAX_FRAME_DIST = 60 * 5 # 5s

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
        print("Processing label file", filename, "...")

        input_labels_filepath = f"output/csv_to_video/{filename}_60fps_dvs_annotation.txt"
        output_labels_filepath = f"output/csv_to_video/{filename}_60fps_dvs_annotation.sep.txt"

        assigned_ids = set()
        max_id = 0
        # find assigned ids and max id
        with open(input_labels_filepath, 'r') as input_labels_file:
            reader = csv.reader(input_labels_file)
            for rrow in reader:
                instance_id = int(rrow[2])
                assigned_ids.add(instance_id)
                if instance_id > max_id:
                    max_id = instance_id
        
        # id:max_frame, ...
        max_frame_index_of_label = {}
        # old_id:new_id, ...
        new_id_for_id = {}

        with open(input_labels_filepath, 'r') as input_labels_file, open(output_labels_filepath, 'w', newline='') as output_labels_file:
            # Read bboxes
            reader = csv.reader(input_labels_file)
            writer = csv.writer(output_labels_file)

            if CSV_HAS_HEADER_ROW:
                rrow = next(reader)
                writer.writerow(rrow)

            for rrow in reader:
                frame_index = int(rrow[0])
                instance_id = int(rrow[2])

                max_frame = max_frame_index_of_label.get(instance_id, None)
                if max_frame is not None:
                    if frame_index - max_frame > MAX_FRAME_DIST:
                        # instances with same id are too far apart -> create new instance
                        new_id = max_id + 1
                        max_id = new_id
                        new_id_for_id[instance_id] = new_id
                        print(f"Assigned new id {new_id} to part of trajectory {instance_id}")
                    else:
                        # replace all coming ids with the new id 
                        # or with the old id, if there is no new id
                        new_id = new_id_for_id.get(instance_id, instance_id)
                else:
                    new_id = instance_id

                wrow = [*rrow]
                wrow[2] = new_id
                writer.writerow(wrow)

                max_frame_index_of_label[instance_id] = frame_index
                


        print(f"Created label file {output_labels_filepath}!")


