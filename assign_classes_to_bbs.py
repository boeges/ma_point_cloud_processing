# Purpose: Load instance_ids and classes from a csv file and assign the (correct) classes
# to an annotation file containing bbs per frame with instance ids and (incorrect) classes.


import csv
from pathlib import Path

def correct_class_name(cla):
    real_class = ""
    if cla=="b" or cla=="bee":
        real_class = "bee"
    elif cla=="u" or cla=="f" or cla=="butterfly":
        real_class = "butterfly"
    elif cla=="d" or cla=="dragonfly":
        real_class = "dragonfly"
    elif cla=="w" or cla=="wasp":
        real_class = "wasp"
    else:
        # Also for case "i"
        real_class = "insect"
    return real_class

############################ MAIN ##################################
if __name__ == "__main__":

    INPUT_LABELS_HAS_HEADER = True
    INPUT_CLASSES_HAS_HEADER = True

    # Paths
    INPUT_LABELS_DIR = Path("output/mu_frame_labels_with_ids")
    INPUT_LABELS_FILENAME = "{filestem}_annotation_instances.csv"

    INPUT_CLASSES_DIR = Path("output/instance_classes")
    INPUT_CLASSES_FILENAME = "{filestem}_instance_classes.csv"

    OUTPUT_LABELS_DIR = Path("output/mu_frame_labels_with_ids")
    OUTPUT_LABELS_FILENAME = "{filestem}_annotation_instances_classes.csv"

    filestems = [
        "1_l-l-l",
        "2_l-h-l",
        "3_m-h-h",
        "4_m-m-h",
        "5_h-l-h",
        "6_h-h-h_filtered",
    ]

    for filestem in filestems:
        input_labels_filepath = INPUT_LABELS_DIR / INPUT_LABELS_FILENAME.format(filestem=filestem)
        input_classes_filepath = INPUT_CLASSES_DIR / INPUT_CLASSES_FILENAME.format(filestem=filestem)
        output_labels_filepath = OUTPUT_LABELS_DIR / OUTPUT_LABELS_FILENAME.format(filestem=filestem)

        if not input_labels_filepath.exists():
            print("Skipping file:", filestem, ". Labels not exist!", input_labels_filepath)
            continue
        if not input_classes_filepath.exists():
            print("Skipping file:", filestem, ". Cclasses file does not exist!", input_classes_filepath)
            continue
        
        print("Processing file:", filestem)

        with open(input_labels_filepath, 'r') as input_labels_file,\
                open(input_classes_filepath, 'r') as input_classes_file,\
                open(output_labels_filepath, 'w', newline='') as output_labels_file:
            labels_reader = csv.reader(input_labels_file)
            classes_reader = csv.reader(input_classes_file)
            labels_writer = csv.writer(output_labels_file)
            
            if INPUT_LABELS_HAS_HEADER:
                next(labels_reader)
            if INPUT_CLASSES_HAS_HEADER:
                next(classes_reader)

            # map the actual classname to each id
            id_class_map = {}
            for row in classes_reader:
                id = int(row[0])
                cla = row[1].lower()
                real_class = correct_class_name(cla)
                id_class_map[id] = real_class

            labels_writer.writerow(["frame_index","class","instance_id","is_difficult","x","y","w","h"])

            for row in labels_reader:
                id = int(row[2])
                orig_class = correct_class_name(row[1])
                new_row = [*row]
                # Get new class or "insect" if there is no class available for this id
                new_class = id_class_map.get(id, "insect")
                if new_class == "insect" and orig_class != "insect" and orig_class != "":
                    # if new class is "insect" and orig class something else, use orig class
                    new_class = orig_class
                new_row[1] = new_class
                labels_writer.writerow(new_row)

        print(f"Saved '{output_labels_filepath}'.")
