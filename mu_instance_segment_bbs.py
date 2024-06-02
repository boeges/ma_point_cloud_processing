# Purpose: Add instance ids to the annotation.csv files from the Münster Dataset.

import csv
import math

class Instance:
    def __init__(self, id, last_x, last_y, last_seen_n_frames_ago):
        self.id = id
        self.last_x = last_x
        self.last_y = last_y
        self.last_seen_n_frames_ago = last_seen_n_frames_ago
        self.is_assigned_this_frame = False

max_bb_dist = 50 # pixels
max_last_seen_n_frames_ago = 100

filenames = [
    "1_l-l-l",
    "2_l-h-l",
    "3_m-h-h",
    "4_m-m-h",
    "5_h-l-h",
    "6_h-h-h_filtered",
]

for filename in filenames:
    print("Processing file:", filename)
    # filename = "1_l-l-l"

    input_csv_filepath = f"../../datasets/muenster_dataset/wacv2024_ictrap_dataset/{filename}_annotation.csv"
    output_csv_filepath = f"output/mu_frame_labels_with_ids/{filename}_annotation_instances.csv"

    highest_used_id = -1
    last_frame_index = "-1"

    # instances while processing
    # [ (<instance_id>, <prev_x_center>, <prev_y_center>, <last_seen_n_frames_ago>), ...]
    open_instances = []

    with open(input_csv_filepath, 'r') as infile, open(output_csv_filepath, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read the header row and add the instance_id header
        header = next(reader)
        header.append('instance_id')
        writer.writerow(["frame_index","class","instance_id","is_difficult","x","y","w","h"])

        for row in reader:
            # check if this row is a new frame
            is_new_frame = last_frame_index != row[0]
            last_frame_index = row[0]

            # instances that will be closed (removed from open_instances)
            close_instances = []

            if is_new_frame:
                # remove instances that havent been seen in a while
                for instance in open_instances:
                    if instance.last_seen_n_frames_ago >= max_last_seen_n_frames_ago:
                        # if <last_seen_n_frames_ago> is too large
                        # add instance to close_instances to remove it from open_instances later
                        close_instances.append(instance.id)
                        continue
                    instance.is_assigned_this_frame = False
                    instance.last_seen_n_frames_ago += 1

            # delete elements by id that are in close_instances
            open_instances = [instance for instance in open_instances if instance.id not in close_instances]

            # row format:
            # frame_index, is_keyframe, class, confidence, left, top, width, height, center_x, center_y
            # 0, True, insect, certain, 1099.8, 566.4, 35.4, 22.2, 1117.5, 577.5
            x, y = float(row[8]), float(row[9])

            # remember which instance is closest and its distance
            closest_instance = None
            min_dist = 9999.0

            # find closest instance that is not already assigned to another bb
            for index, instance in enumerate(open_instances):
                if not instance.is_assigned_this_frame:
                    dist = math.sqrt((instance.last_x - x)**2 + (instance.last_y - y)**2)
                    if dist < max_bb_dist and dist < min_dist:
                        min_dist = dist
                        closest_instance = instance

            if closest_instance is None:
                # create new instance id
                highest_used_id += 1
                closest_instance = Instance(highest_used_id, x, y, 0)
                open_instances.append(closest_instance)
                closest_instance_index = len(open_instances)-1

            closest_instance.is_assigned_this_frame = True
            closest_instance.last_seen_n_frames_ago = 0
            closest_instance.last_x = x
            closest_instance.last_y = y
            
            # add new column to row
            row.append(closest_instance.id)
            
            # Write row in new format
            # old format: frame_index, is_keyframe, class, confidence, left,top,width,height,center_x,center_y
            # new format: "frame_index","class","instance_id","is_difficult","x","y","w","h"
            new_row = [row[0], row[2], closest_instance.id, int(row[3]=="0"), round(float(row[4])), round(float(row[5])), round(float(row[6])), round(float(row[7]))]
            writer.writerow(new_row)

    print(f"Saved '{output_csv_filepath}'.")
