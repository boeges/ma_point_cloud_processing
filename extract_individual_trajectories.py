# Purpose: Extract individual insect flight paths.
# For each path create a separate csv file containing the (x, y, t)-Points.

# Format for events: (x, y, t_in_us_or_ms, p)
# Format for Labels from DarkLabel software: (frame_index, classname, instance_id, is_difficult, x, y, w, h) 

import csv
from datetime import datetime
from pathlib import Path
import bee_utils as bee


# a single bbox
class Label:
    def __init__(self, left, top, width, height, instance_id, is_confident):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.instance_id = instance_id
        self.is_confident = is_confident
        self._current_bb_index = 0

# contains multiple labels for a frame
class Frame:
    def __init__(self, start, index):
        self.start = start
        self.index = index
        self.labels = {}

class InstanceTrajectory:
    def __init__(self):
        self.first_timestamp = None
        self.cla = None # class ("bee", "butterfly", "dragonfly", "wasp")
        self.events = []


def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


############################ MAIN ##################################
if __name__ == "__main__":
    DATETIME_STR = getDateTimeStr()

    PRINT_PROGRESS_EVERY_N_PERCENT = 1

    EVENT_COLOR = (255,255,255)
    UNCONFIDENT_COLOR = (255,153,153)
    BBOX_COLORS = [
        (127,255,212),
        (255,127,80),
        (255,0,255),
        (0,128,0)
    ]

    # to sync labels and events. ideally 0
    # e.g. "1000" means the first frame is 1000us (or ms) delayed
    # FRAME_TIME_OFFSET = TIMESTEPS_PER_FRAME * 0.5
    FRAME_TIME_OFFSET = 0
    # Create subframes between frames with interpolated bboxes
    # 0: No subframes
    # 1: Create one subframe between two real frames (doubles frame count)
    SUBFRAME_COUNT = 1

    CREATE_BBOX_EVENTS = False
    ADD_DATETIME_TO_OUTPUT_DIR = True
    DATETIME_PREFIX = f"_{DATETIME_STR}" if ADD_DATETIME_TO_OUTPUT_DIR else ""

    FRAMETIMES_CSV_DIR = None # Overwrite this with a path if used!

   ############### PF #############
    # FPS = 60

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

    # # Format for events: (x, y, t (us or ms), p)
    # EVENTS_CSV_HAS_HEADER = False
    # EVENTS_CSV_X_COL = 0
    # EVENTS_CSV_Y_COL = 1
    # EVENTS_CSV_T_COL = 2
    # EVENTS_CSV_P_COL = 3
    
    # FRAMETIMES_FILE_HAS_HEADER = True

    # # 2_separated, 3_classified
    # SEGMENTATION_STAGE = "3_classified"

    # # Paths
    # EVENTS_CSV_DIR = Path("../../datasets/Insektenklassifikation")
    # EVENTS_CSV_FILENAME = "{filestem}.csv"

    # LABELS_CSV_DIR = Path("output/video_annotations") / SEGMENTATION_STAGE
    # LABELS_CSV_FILENAME = "{filestem}.csv"

    # FRAMETIMES_CSV_DIR = Path("output/video_frametimes")
    # FRAMETIMES_CSV_FILENAME = "{filestem}_60fps_dvs_frametimes_v1.csv"

    # OUTPUT_BASE_DIR = Path("output/extracted_trajectories") /  f"{SEGMENTATION_STAGE}{DATETIME_PREFIX}"

    ##################################

    ############### MU #############
    # FPS = 100

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

    # # Format for events: (x, y, t (us), p)
    # EVENTS_CSV_HAS_HEADER = True
    # EVENTS_CSV_X_COL = 0
    # EVENTS_CSV_Y_COL = 1
    # EVENTS_CSV_T_COL = 2
    # EVENTS_CSV_P_COL = 3

    # FRAMETIMES_FILE_HAS_HEADER = True

    # # 2_separated, 3_classified
    # SEGMENTATION_STAGE = "2_separated"

    # # Paths
    # EVENTS_CSV_DIR = Path("output/mu_h5_to_csv")
    # EVENTS_CSV_FILENAME = "{filestem}.csv"

    # LABELS_CSV_DIR = Path("output/video_annotations") / SEGMENTATION_STAGE
    # LABELS_CSV_FILENAME = "{filestem}.csv"

    # FRAMETIMES_CSV_DIR = Path("output/mu_h5_frametimes_to_csv")
    # FRAMETIMES_CSV_FILENAME = "{filestem}_frametimes.csv"

    # OUTPUT_BASE_DIR = Path("output/extracted_trajectories") / f"{SEGMENTATION_STAGE}{DATETIME_PREFIX}"

    ##################################

    ############### MB #############
    FPS = 60

    # Format from DarkLabel software: (frame_index, classname, instance_id, is_difficult, x, y, w, h) 
    #                                      0            1          2            3         4  5  6  7
    LABELS_CSV_HAS_HEADER = False
    LABELS_CSV_FRAME_COL = 0 # frame index
    LABELS_CSV_CLASS_COL = 1 # class name
    LABELS_CSV_ID_COL = 2 # instance id
    LABELS_CSV_IS_CONFIDENT_COL = 3
    LABELS_CSV_X_COL = 4
    LABELS_CSV_Y_COL = 5
    LABELS_CSV_W_COL = 6
    LABELS_CSV_H_COL = 7
    BB_IS_CONFIDENT_WHEN_MATCHES = "0" # is_difficult: 0(=confident)/1(=difficult)

    # Format for events: (x, y, t (us or ms), p)
    EVENTS_CSV_HAS_HEADER = False
    EVENTS_CSV_X_COL = 0
    EVENTS_CSV_Y_COL = 1
    EVENTS_CSV_T_COL = 3
    EVENTS_CSV_P_COL = 2
    
    FRAMETIMES_FILE_HAS_HEADER = True

    # 2_separated, 3_classified
    SEGMENTATION_STAGE = "3_classified"

    # Paths
    EVENTS_CSV_DIR = Path("../../aufnahmen/2024-08-06_libellen/exported_csv/") #  2024-07-18_bunter_garten, 2024-08-06_libellen
    EVENTS_CSV_FILENAME = "{filestem}.csv"

    LABELS_CSV_DIR = Path("output/video_annotations") / SEGMENTATION_STAGE
    LABELS_CSV_FILENAME = "{filestem}.csv"

    OUTPUT_BASE_DIR = Path("output/extracted_trajectories") /  f"{SEGMENTATION_STAGE}{DATETIME_PREFIX}"

    ##################################


    WIDTH = 1280
    HEIGHT = 720
    # Precision of the timestamp: for mikroseconds: 1000000, for milliseconds: 1000
    TIMESTEPS_PER_SECOND = 1_000_000
    # If timestamp in mikroseconds: -> mikroseconds per frame
    TIMESTEPS_PER_FRAME = (1 / FPS) * TIMESTEPS_PER_SECOND
    HALF_FRAME_TIME = TIMESTEPS_PER_FRAME // 2
    T_SCALE = 0.002 # 0.002 is good
    

    # Find all csv files in EVENTS_CSV_DIR
    events_filepaths = [file for file in EVENTS_CSV_DIR.iterdir() if (file.is_file() and file.name.endswith(".csv"))]
    print(f"Found {len(events_filepaths)} csv files containing events")

    # Files that use the old/defect v1 video export format (wrong frametimes!).
    # They need an extra file with frame_index to timestamp mapping
    v1_video_filestems = [
        "hauptsächlichBienen1",
        "vieleSchmetterlinge2",
        "1_l-l-l",
        "2_l-h-l",
        "3_m-h-h",
        "4_m-m-h",
        "5_h-l-h",
        "6_h-h-h_filtered",
    ]

    # Scenes
    filestems = [
        # PF
        # "hauptsächlichBienen1",
        # "hauptsächlichBienen2",
        # "libellen1",
        # "libellen2",
        # "libellen3",
        # "vieleSchmetterlinge1",
        # "vieleSchmetterlinge2",
        # "wespen1",
        # "wespen2",
        # "wespen3",
        # MU
        # "1_l-l-l",
        # "2_l-h-l",
        # "3_m-h-h",
        # "4_m-m-h",
        # "5_h-l-h",
        # "6_h-h-h_filtered",
        # MB
        # "mb-bum2-2",
        # "mb-dra1-1",
        "mb-dra2-1",
    ]

    print("Using t-scale:", T_SCALE)

    for events_filepath in events_filepaths:
        filestem = events_filepath.name.replace(".csv", "")
        labels_filepath = LABELS_CSV_DIR / LABELS_CSV_FILENAME.format(filestem=filestem)

        if not labels_filepath.exists():
            print("#### Skipping file", events_filepath.name, ". Labels file does not exist:", labels_filepath.name)
            continue

        if len(filestems) > 0 and filestem not in filestems:
            print("#### Skipping file:", events_filepath.name, ". Not in accepted filestems.")
            continue

        print("#### Processing file:", filestem, "####")

        # Read frametimes from csv
        frametimes = []
        read_frametimes_from_file = False # False for v3 videos and annotations; True for v1 videos and annotations
        if filestem in v1_video_filestems:
            read_frametimes_from_file = True
            frametimes_filepath = FRAMETIMES_CSV_DIR / FRAMETIMES_CSV_FILENAME.format(filestem=filestem)
            print("Reading frametimes from file:", frametimes_filepath.name)
            with open(frametimes_filepath, 'r', ) as frametimes_file:
                frametimes_reader = csv.reader(frametimes_file)

                if FRAMETIMES_FILE_HAS_HEADER:
                    # skip header
                    frametimes_row = next(frametimes_reader, None)
                
                # row (frame_index, timestamp)
                for frametimes_row in frametimes_reader:
                    frametimes.append(float(frametimes_row[1]))
        else:
            print("NOT reading frametimes from file!")


        # Contains all labels per frame
        # old [ Frame{start, index, labels:[ Label{is_confident, left, top, width, height, instance_id}, ... ]}, ... ]
        # new [ Frame{start, index, labels: { <instance_id> : Label{is_confident, left, top, width, height, instance_id}, ... } }, ... ]
        frames = []

        frame_time = 0
        frame_index = 0

        # Find unique instance ids and store all events for each instance
        # { <instance_id>: InstanceTrajectory{ first_timestamp, cla, events:[ (x, y, t, is_confident, r, g, b), ... ] }, ... }
        trajectories = {}
        
        with open(labels_filepath, 'r') as labels_file, open(events_filepath, 'r') as events_file:
            # Read labels
            labels_reader = csv.reader(labels_file)
            if LABELS_CSV_HAS_HEADER:
                labels_row = next(labels_reader, None)
            labels_row = next(labels_reader, None)

            print("Finding timestamps, bboxes for frames and unique instance ids")


            while labels_row is not None:
                corrected_frame_time = frametimes[frame_index] if read_frametimes_from_file else (frame_time + FRAME_TIME_OFFSET)
                frame = Frame(corrected_frame_time, frame_index)
                frames.append(frame)

                # there could be multiple bboxes in one frame index
                while labels_row is not None:
                    label_frame_index = int(labels_row[LABELS_CSV_FRAME_COL])

                    if label_frame_index > frame_index:
                        # If the label frame index is for a future video frame
                        break
                    if label_frame_index == frame_index:
                        # If label frame index matches video frame index:
                        # Add bbox to frame bboxes
                        # row format (frame_index, classname, instance_id, is_difficult, x, y, w, h)
                        instance_id = int(labels_row[LABELS_CSV_ID_COL])
                        label = Label(
                            int(float(labels_row[LABELS_CSV_X_COL])),
                            int(float(labels_row[LABELS_CSV_Y_COL])),
                            int(float(labels_row[LABELS_CSV_W_COL])),
                            int(float(labels_row[LABELS_CSV_H_COL])),
                            instance_id,
                            labels_row[LABELS_CSV_IS_CONFIDENT_COL] == BB_IS_CONFIDENT_WHEN_MATCHES
                        )
                        frame.labels[instance_id] = label
                        if instance_id not in trajectories:
                            tra = InstanceTrajectory()
                            cla = labels_row[LABELS_CSV_CLASS_COL]
                            tra.cla = bee.full_class_to_short_class(cla, "ins")
                            trajectories[instance_id] = tra
                    else:
                        # label_frame_index < current_frame_index
                        # Iterate until matching frame has been found
                        pass
                    # read next row
                    labels_row = next(labels_reader, None)

                frame_index += 1
                frame_time += TIMESTEPS_PER_FRAME

            print("Processed", len(frames), "frames. First 3 frames:")
            for i, frame in enumerate(frames[:3]):
                print(f"Frame {i} has {len(frame.labels)} labels, index={frame.index}, start={frame.start}")

            print("First 3 frames containing labels")
            j = 0
            for i, frame in enumerate(frames):
                if len(frame.labels) != 0:
                    print(f"Frame {i} has {len(frame.labels)} labels, index={frame.index}, start={frame.start}")
                    j+=1
                    if j >= 3:
                        break


            if SUBFRAME_COUNT == 0:
                # Dont create subframes
                interp_frames = frames
            else:
                # Create subframes between frames with interpolated bboxes
                frames_iter = iter(frames)
                prev_frame = next(frames_iter, None)

                # interpolate a bbox between last_frame and frame at constant intervals
                
                subframe = Frame(prev_frame.start, 0)
                subframe.labels = prev_frame.labels
                interp_frames = [ subframe ] # interpolated frames; Contains real and subframes

                for frame in frames_iter:
                    for subframe_index in range(SUBFRAME_COUNT):
                        # pos: Position between the two frames [0.0, 1.0]
                        pos = (subframe_index + 1) / (SUBFRAME_COUNT + 1)
                        subframe = Frame(prev_frame.start + (TIMESTEPS_PER_FRAME * pos), len(interp_frames))
                        interp_frames.append(subframe)
                        for instance_id, prev_label in prev_frame.labels.items():
                            label = frame.labels.get(instance_id, None)
                            if label is not None:
                                subframe_label = Label(
                                    int(prev_label.left * pos + label.left * (1.0-pos)),
                                    int(prev_label.top * pos + label.top * (1.0-pos)),
                                    int(prev_label.width * pos + label.width * (1.0-pos)),
                                    int(prev_label.height * pos + label.height * (1.0-pos)),
                                    instance_id,
                                    label.is_confident
                                )
                                subframe.labels[instance_id] = subframe_label

                    subframe = Frame(frame.start, len(interp_frames))
                    subframe.labels = frame.labels
                    interp_frames.append( subframe )
                    prev_frame = frame
                    

            frames = interp_frames
            print("Interpolated frames:", len(frames), " (real + subframes)")
           
            # Find total row count for tracking progress
            print("Scanning events CSV file to find total row count...")
            event_row_count = sum(1 for _ in events_file)

            # Reader for events file
            events_file.seek(0)
            event_reader = csv.reader(events_file)
            if EVENTS_CSV_HAS_HEADER:
                event_row = next(event_reader, None)

            event_row = next(event_reader, None)

            # get start timestamp of first frame; Loop will start with second frame
            frames_iter = iter(frames)
            prev_frame = next(frames_iter, None)

            last_progress_percent = 0

            print(f"Event CSV contains {event_row_count} rows. Starting scanning file and assigning events to insect instances...")
            print("Progress (%): ", sep="", end="")

            # iterate frames; inside, iterate events matching the frame
            for i,frame in enumerate(frames_iter):
                # find matching frame from "frames".
                # find in which bb an event is.

                # DEBUG
                # if frame.index > 100:
                #     break

                # print progress
                progress_percent = int(i / len(frames) * 100)
                if progress_percent >= last_progress_percent + 1:
                    last_progress_percent = progress_percent
                    print(progress_percent, ", ", sep="", end="")

                # many events can match th
                # event structure: (x, y, p, t) or (x, y, t, p)
                while event_row is not None:
                    event_x, event_y = int(event_row[EVENTS_CSV_X_COL]), int(event_row[EVENTS_CSV_Y_COL])
                    event_timestamp = float(event_row[EVENTS_CSV_T_COL])

                    # Is event between prev_frame and frame?
                    # (Subtract HALF_FRAME_TIME from frame start time, so that the time-center of each
                    # bbox is at its frame start time.) -> doesnt work well
                    if event_timestamp >= frame.start: # or better (frame.start + HALF_FRAME_TIME) ?
                        # event is in next frame!
                        break

                    progress_percent = int(i / len(frames) * 100)
                    if progress_percent >= last_progress_percent + 1:
                        last_progress_percent = progress_percent
                        print(progress_percent, ", ", sep="", end="")

                    # event is in previous frame
                    # is event in a bbox?
                    for instance_id,label in prev_frame.labels.items():
                        # get trajectory for this instance
                        instance_trajectory = trajectories[instance_id]

                        if event_x >= label.left and \
                                event_x < label.left + label.width and \
                                event_y >= label.top and \
                                event_y < label.top + label.height:
                            # -> event is in this bbox
                            
                            if instance_trajectory.first_timestamp is None:
                                instance_trajectory.first_timestamp = event_timestamp

                            # shift (to zero) and scale timestamp coordinate
                            event_t_scaled = (event_timestamp - instance_trajectory.first_timestamp) * T_SCALE
                            is_confident = 1 if label.is_confident else 0
                            event_color = EVENT_COLOR if label.is_confident else UNCONFIDENT_COLOR
                            instance_trajectory.events.append( (event_x, event_y, event_t_scaled, *event_color, is_confident, prev_frame.index, -1) )

                    event_row = next(event_reader, None)

                # For DEBUG: Draw bbox corners as events
                if CREATE_BBOX_EVENTS:
                    for instance_id,label in prev_frame.labels.items():
                        # get trajectory for this instance
                        instance_trajectory = trajectories[instance_id]
                        if instance_trajectory.first_timestamp is not None:
                            # DEBUG: Add pints of bbox
                            points_color = BBOX_COLORS[label._current_bb_index % 3]
                            is_confident = 0

                            # at start of frame
                            frame_start = ((prev_frame.start - instance_trajectory.first_timestamp) * T_SCALE)
                            instance_trajectory.events.append( (label.left, label.top, frame_start, *points_color, is_confident, prev_frame.index, 0b000) )
                            instance_trajectory.events.append( (label.left+label.width, label.top, frame_start, *points_color, is_confident, prev_frame.index, 0b001) )
                            instance_trajectory.events.append( (label.left, label.top+label.height, frame_start, *points_color, is_confident, prev_frame.index, 0b010) )
                            instance_trajectory.events.append( (label.left+label.width, label.top+label.height, frame_start, *points_color, is_confident, prev_frame.index, 0b011) )
                            # at end of frame (start of next frame)
                            frame_end = ((frame.start - instance_trajectory.first_timestamp) * T_SCALE)
                            instance_trajectory.events.append( (label.left, label.top, frame_end, *points_color, is_confident, prev_frame.index, 0b100) )
                            instance_trajectory.events.append( (label.left+label.width, label.top, frame_end,*points_color, is_confident, prev_frame.index, 0b101) )
                            instance_trajectory.events.append( (label.left, label.top+label.height, frame_end, *points_color, is_confident, prev_frame.index, 0b110) )
                            instance_trajectory.events.append( (label.left+label.width, label.top+label.height, frame_end, *points_color, is_confident, prev_frame.index, 0b111) )

                            label._current_bb_index += 1

                prev_frame = frame

            # print debug stuff
            print("\nNumber of instance trajectories:", len(trajectories))
            for id, trajectory in trajectories.items():
                print("trajectory", id, "has", len(trajectory.events), "events and starts at t =", trajectory.first_timestamp)

            # Save trajectories        
            print("Saving trajectories...")
            
            # Write CSVs
            for id, trajectory in trajectories.items():
                if len(trajectory.events) == 0:
                    print("Skipping trajectory", id, ". No points!")
                    continue
                
                output_dir_path = (OUTPUT_BASE_DIR / "_with_bboxes") if CREATE_BBOX_EVENTS else OUTPUT_BASE_DIR
                output_dir_path = output_dir_path / f"{filestem}_trajectories"
                output_dir_path.mkdir(parents=True, exist_ok=True)

                output_file_path = output_dir_path / f"{id}_{trajectory.cla}_pts{len(trajectory.events)}_start{int(trajectory.first_timestamp)}.csv"

                with open(output_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    # bb_corner_index: -1: normal event; 0-7 corner indices as 0b000 with zyx order; 0=in front; 1=behind
                    writer.writerow(["x", "y", "t", "r", "g", "b", "is_confident", "bb_frame_index", "bb_corner_index"])
                    writer.writerows(trajectory.events)

                print(f"Created {output_file_path}!")

    print("Finished!")

                













