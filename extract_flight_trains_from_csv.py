# Purpose: Extract individual insect flight paths.
# For each path create a separate csv file containing the (x, y, t)-Points.

# Format for events: (x, y, t_in_us_or_ms, p)
# Format from DarkLabel software: (frame_index, classname, instance_id, is_difficult, x, y, w, h) 

import csv
from datetime import datetime
from pathlib import Path


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
        self.events = []


def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



############################ MAIN ##################################
if __name__ == "__main__":
    datetime_str = getDateTimeStr()

    WIDTH = 1280
    HEIGHT = 720
    FPS = 60
    # Precision of the timestamp: for mikroseconds: 1000000, for milliseconds: 1000
    TIMESTEPS_PER_SECOND = 1_000_000
    # If timestamp in mikroseconds: -> mikroseconds per frame
    TIMESTEPS_PER_FRAME = (1 / FPS) * TIMESTEPS_PER_SECOND
    HALF_FRAME_TIME = TIMESTEPS_PER_FRAME // 2
    T_SCALE = 0.002 # 0.002 is good
    EVENTS_CSV_HAS_HEADER = False
    LABELS_CSV_HAS_HEADER = False

    # Format for events: (x, y, t (us or ms), p)
    EVENTS_CSV_X_COL = 0
    EVENTS_CSV_Y_COL = 1
    EVENTS_CSV_T_COL = 2
    EVENTS_CSV_P_COL = 3

    #                                      0            1          2            3         4  5  6  7
    # Format from DarkLabel software: (frame_index, classname, instance_id, is_difficult, x, y, w, h) 
    LABELS_CSV_FRAME_COL = 0 # frame index
    LABELS_CSV_CLASS_COL = 1 # class name
    LABELS_CSV_ID_COL = 2 # instance id
    LABELS_CSV_IS_CONFIDENT_COL = 3 # frame index
    LABELS_CSV_X_COL = 4
    LABELS_CSV_Y_COL = 5
    LABELS_CSV_W_COL = 6
    LABELS_CSV_H_COL = 7

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
    SUBFRAME_COUNT = 0
    CREATE_BBOX_EVENTS = True
    ADD_TIME_TO_FILENAME = True
    READ_FRAMETIMES_FROM_FILE = True
    FRAMETIMES_FILE_HAS_HEADER = True


    filenames = [
        # "hauptsächlichBienen1",
        # "hauptsächlichBienen2",
        # "libellen1",
        # "libellen2",
        # "libellen3",
        # "vieleSchmetterlinge1",
        "vieleSchmetterlinge2",
    ]

    print("Using t-scale:", T_SCALE)

    for filename in filenames:
        print("Processing file:", filename)

        events_filepath = f"../../datasets/Insektenklassifikation/{filename}.csv"
        labels_filepath = f"output/csv_to_video/{filename}_60fps_dvs_annotation.sep.txt"
        output_base_dir = "output/"

        frametimes_filepath = f"output/csv_to_video/{filename}_60fps_dvs_frametimes_v1.csv"
        frametimes = []

        if READ_FRAMETIMES_FROM_FILE:
            with open(frametimes_filepath, 'r') as frametimes_file:
                frametimes_reader = csv.reader(frametimes_file)

                if FRAMETIMES_FILE_HAS_HEADER:
                    # skip header
                    frametimes_row = next(frametimes_reader, None)
                
                # row (frame_index, timestamp)
                for frametimes_row in frametimes_reader:
                    frametimes.append(float(frametimes_row[1]))



        # Contains all labels per frame
        # old [ Frame{start, index, labels:[ Label{is_confident, left, top, width, height, instance_id}, ... ]}, ... ]
        # new [ Frame{start, index, labels: { <instance_id> : Label{is_confident, left, top, width, height, instance_id}, ... } }, ... ]
        frames = []

        frame_time = 0
        frame_index = 0
        
        with open(labels_filepath, 'r') as labels_file, open(events_filepath, 'r') as events_file:
            # Read labels
            labels_reader = csv.reader(labels_file)
            if LABELS_CSV_HAS_HEADER:
                labels_row = next(labels_reader, None)
            labels_row = next(labels_reader, None)

            print("Finding timestamps and bboxes for frames...")

            while labels_row is not None:
                if READ_FRAMETIMES_FROM_FILE:
                    corrected_frame_time = frametimes[frame_index]
                else:
                    corrected_frame_time = frame_time + FRAME_TIME_OFFSET

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
                        label = Label(
                            int(float(labels_row[LABELS_CSV_X_COL])),
                            int(float(labels_row[LABELS_CSV_Y_COL])),
                            int(float(labels_row[LABELS_CSV_W_COL])),
                            int(float(labels_row[LABELS_CSV_H_COL])),
                            int(labels_row[LABELS_CSV_ID_COL]),
                            int(labels_row[LABELS_CSV_IS_CONFIDENT_COL]) == 0
                        )
                        frame.labels[int(labels_row[LABELS_CSV_ID_COL])] = label
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

            # store all events for each instance
            # { <instance_id>: InstanceEvents{ first_timestamp, events:[ (x, y, t, is_confident, r, g, b), ... ] }
            trajectories = {}

            # Find total row count for tracking progress
            print("Scanning events CSV file to find total row count...")
            event_row_count = sum(1 for _ in events_file)
            
            # Iterate events file
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
                        if event_x >= label.left and \
                                event_x < label.left + label.width and \
                                event_y >= label.top and \
                                event_y < label.top + label.height:
                            # -> event is in this bbox

                            # create or get events for this instance
                            instance_trajectory = trajectories.setdefault(instance_id, InstanceTrajectory())
                            if instance_trajectory.first_timestamp is None:
                                instance_trajectory.first_timestamp = event_timestamp

                            # shift (to zero) and scale timestamp coordinate
                            event_z = (event_timestamp - instance_trajectory.first_timestamp) * T_SCALE
                            is_confident = 1 if label.is_confident else 0
                            event_color = EVENT_COLOR if label.is_confident else UNCONFIDENT_COLOR
                            instance_trajectory.events.append( (event_x, event_y, event_z, is_confident, *event_color) )

                            if CREATE_BBOX_EVENTS:
                                # DEBUG: Add pints of bbox
                                # at start of frame
                                points_color = BBOX_COLORS[label._current_bb_index % 3]

                                frame_start = ((prev_frame.start - instance_trajectory.first_timestamp) * T_SCALE)
                                instance_trajectory.events.append( (label.left, label.top, frame_start, is_confident, *points_color) )
                                instance_trajectory.events.append( (label.left+label.width, label.top, frame_start, is_confident, *points_color) )
                                instance_trajectory.events.append( (label.left, label.top+label.height, frame_start, is_confident, *points_color) )
                                instance_trajectory.events.append( (label.left+label.width, label.top+label.height, frame_start, is_confident, *points_color) )
                                # at end of frame (start of next frame)
                                frame_end = ((frame.start - instance_trajectory.first_timestamp) * T_SCALE)
                                instance_trajectory.events.append( (label.left, label.top, frame_end, is_confident, *points_color) )
                                instance_trajectory.events.append( (label.left+label.width, label.top, frame_end,is_confident, *points_color) )
                                instance_trajectory.events.append( (label.left, label.top+label.height, frame_end, is_confident, *points_color) )
                                instance_trajectory.events.append( (label.left+label.width, label.top+label.height, frame_end, is_confident, *points_color) )

                                label._current_bb_index += 1

                    event_row = next(event_reader, None)

                prev_frame = frame

            # print debug stuff
            print("Finished assigning events")
            print("Number of instance trajectories:", len(trajectories))
            for id, trajectory in trajectories.items():
                print("trajectory", id, "has", len(trajectory.events), "events and starts at t =", trajectory.first_timestamp)

            # Save trajectories        
            print("Saving trajectories...")
            
            # Write CSVs
            for id, trajectory in trajectories.items():
                if len(trajectory.events) == 0:
                    print("Skipping trajectory", id, ". No points!")
                    continue
                
                bbox_filename_part = "_bbox" if CREATE_BBOX_EVENTS else ""
                datetime_filename_part = ("_"+datetime_str) if ADD_TIME_TO_FILENAME else ""

                output_dir_path = Path(output_base_dir) / f"{filename}_trajectories{bbox_filename_part}{datetime_filename_part}"
                output_dir_path.mkdir(parents=True, exist_ok=True)

                output_file_path = output_dir_path / f"{id}.csv"

                with open(output_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["x", "y", "t", "is_confident", "r", "g", "b"])
                    writer.writerows(trajectory.events)

                print(f"Created {output_file_path}!")

    print("Finished!")



                













