# Purpose: Load instance_ids and classes from a csv file and assign the (correct) classes
# to an annotation file containing bbs per frame with instance ids and (incorrect) classes.


import csv
from pathlib import Path
from bee_utils import parse_full_class_name

############################ MAIN ##################################
if __name__ == "__main__":

    INPUT_LABELS_HAS_HEADER = True
    INPUT_CLASSES_HAS_HEADER = True

    # Paths
    LABELS_CSV_DIR = Path("output/video_annotations/2_separated")
    LABELS_CSV_FILENAME = "{filestem}.csv"
    # LABELS_CSV_FILENAME = "{filestem}_60fps_dvs_sep.txt"

    INPUT_CLASSES_DIR = Path("output/instance_classes")
    INPUT_CLASSES_FILENAME = "{filestem}.csv"

    OUTPUT_LABELS_DIR = Path("output/video_annotations/3_classified")
    OUTPUT_LABELS_FILENAME = "{filestem}.csv"

    # Find all csv files in EVENTS_CSV_DIR
    labels_filepaths = [file for file in LABELS_CSV_DIR.iterdir() if file.is_file()]
    print(f"Found {len(labels_filepaths)} csv files containing labels")

    for labels_filepath in labels_filepaths:
        print(labels_filepath.name)

    # exit()

    # filestems = [
    #     "1_l-l-l",
    #     "2_l-h-l",
    #     "3_m-h-h",
    #     "4_m-m-h",
    #     "5_h-l-h",
    #     "6_h-h-h_filtered",
    # ]

    for labels_filepath in labels_filepaths:
        filestem = labels_filepath.name.replace(LABELS_CSV_FILENAME.format(filestem=""), "")

        classes_filepath = INPUT_CLASSES_DIR / INPUT_CLASSES_FILENAME.format(filestem=filestem)
        output_labels_filepath = OUTPUT_LABELS_DIR / OUTPUT_LABELS_FILENAME.format(filestem=filestem)

        if not classes_filepath.exists():
            print("Skipping file:", labels_filepath.name, "; Classes file", classes_filepath.name, "does not exist!")
            continue
            
        print("\nProcessing file:", labels_filepath.name)

        id_class_map = {}
        with open(classes_filepath, 'r') as classes_file:
            classes_reader = csv.reader(classes_file)
            if INPUT_CLASSES_HAS_HEADER:
                next(classes_reader)
            # map the actual classname to each id
            for row in classes_reader:
                id = int(row[0])
                cla = row[1].lower()
                real_class = parse_full_class_name(cla, "insect")
                id_class_map[id] = real_class

        with open(labels_filepath, 'r') as labels_file,\
                open(output_labels_filepath, 'w', newline='') as output_labels_file:
            labels_reader = csv.reader(labels_file)
            labels_writer = csv.writer(output_labels_file)
            
            if INPUT_LABELS_HAS_HEADER:
                next(labels_reader)

            # labels_writer.writerow(["frame_index","class","instance_id","is_difficult","x","y","w","h"])

            for row in labels_reader:
                id = int(row[2])
                orig_class = parse_full_class_name(row[1], "insect")
                new_row = [*row]
                # Get new class or "insect" if there is no class available for this id
                new_class = id_class_map.get(id, "insect")
                if new_class == "insect" and orig_class != "insect" and orig_class != "":
                    # if new class is "insect" and orig class something else, use orig class
                    new_class = orig_class
                new_row[1] = new_class
                labels_writer.writerow(new_row)

        print(f"Saved '{output_labels_filepath}'.")
