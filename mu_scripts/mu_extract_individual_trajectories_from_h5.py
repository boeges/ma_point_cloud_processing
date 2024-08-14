# Purpose: Extract individual insect flight paths.
# For each path create a separate csv file containing the (x, y, t)-Points.

# DEPRECATED!
# Use mu_h5_events_to_csv.py, mu_h5_frametimes_to_csv.py and extract_individual_trajectories_from_csv.py instead!

import csv
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path


# a single bbox
class Label:
    def __init__(self, left, top, height, width, instance_id, is_confident):
        self.left = left
        self.top = top
        self.height = height
        self.width = width
        self.instance_id = instance_id
        self.is_confident = is_confident

# contains multiple labels for a frame
class Frame:
    def __init__(self, start, index):
        self.start = start
        self.index = index
        self.labels = []

class InstanceTrajectory:
    def __init__(self):
        self.first_timestamp = None
        self.events = []


def getDateTimeStr():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



############################ MAIN ##################################
if __name__ == "__main__":
    datetime_str = getDateTimeStr()

    width = 1280
    height = 720
    t_scale = 0.002 # 0.002 is good

    print("Using t scale:", t_scale)

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

        events_filepath = f"../../datasets/muenster_dataset/wacv2024_ictrap_dataset/{filename}.h5"
        labels_filepath = f"output/{filename}_annotation_instances.csv"
        output_base_dir = "output/"
        # csv_filepath = f"{filename}_{getDateTimeStr()}.csv"

        # Contains all labels per frame
        # [ Frame{start, index, labels:[ Label{is_confident, left, top, width, height, instance_id}, ... ]}, ... ]
        frames = []

        # event zB (133, 716, 1, 1475064)
        with h5py.File(events_filepath, "r") as events_file, open(labels_filepath, 'r') as cvs_file:
            # Read events and triggers
            events = events_file["CD/events"] # event (x, y, p, t) (133, 716, 1, 1475064)
            triggers = events_file["EXTERNAL_TRIGGERS/corrected_positive"] # (p, t, channel_id, frame_index)

            first_event_timestamp = events[0][3]

            # Read labels
            reader = csv.DictReader(cvs_file)
            label_row = next(reader, None)

            print("Finding timestamps and bboxes for frames...")

            for trigger in triggers:
                trigger_t = int(trigger[1])
                trigger_frame_index = int(trigger[3])

                frame = Frame(trigger_t, trigger_frame_index)
                frames.append(frame)


                # there could be multiple rows for one frame index
                while label_row is not None:
                    label_frame_index = int(label_row["frame_index"])

                    if label_frame_index > trigger_frame_index:
                        # If the label frame index is for a future trigger
                        break
                    if label_frame_index == trigger_frame_index:
                        # If label frame index matches video frame index:
                        # Add bbox to frame bboxes
                        label = Label(
                            int(float(label_row["left"])),
                            int(float(label_row["top"])),
                            int(float(label_row["height"])),
                            int(float(label_row["width"])),
                            int(label_row["instance_id"]),
                            label_row["confidence"] == "certain"
                        )
                        frame.labels.append(label)
                    else:
                        # label_frame_index < trigger_frame_index
                        # Iterate until matching frame has been found
                        pass
                    # read next row
                    label_row = next(reader, None)

            print("Processed", len(frames), "frames. First 3 frames:")
            for i, frame in enumerate(frames[:3]):
                print(f"Frame {i} {len(frame.labels)} has labels, index={frame.index}, start={frame.start}")

            print("Assigning events to insect instances...")

            # store all events for each instance
            # { <instance_id>: InstanceEvents{ first_timestamp, events:[ (x, y, t, is_confident), ... ] }
            trajectories = {}

            events_iter = iter(events)
            event = next(events_iter)

            # get start timestamp of first frame; Loop will start with second frame
            frames_iter = iter(frames)
            prev_frame = next(frames_iter, None)

            last_progress_percent = 0

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
                # event structure: (x, y, p, t)
                while event is not None:
                    event_x, event_y = event[0], event[1]
                    event_timestamp = event[3]

                    # is event between prev_frame and frame?
                    if event_timestamp >= frame.start:
                        # event is in next frame!
                        break

                    # event is in previous frame
                    # is event in a bbox?
                    for label in prev_frame.labels:
                        if event_x >= label.left and \
                                event_x < label.left + label.width and \
                                event_y >= label.top and \
                                event_y < label.top + label.height:
                            # -> event is in this bbox
                            # create or get events for this instance
                            instance_trajectory = trajectories.setdefault(label.instance_id, InstanceTrajectory())
                            if instance_trajectory.first_timestamp is None:
                                instance_trajectory.first_timestamp = event_timestamp
                            # shift (to zero) and scale timestamp coordinate
                            event_z = (event_timestamp - instance_trajectory.first_timestamp) * t_scale
                            instance_trajectory.events.append( (event_x, event_y, event_z, (1 if label.is_confident else 0)) )

                    event = next(events_iter, None)

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

                output_dir_path = Path(output_base_dir) / f"{filename}_trajectories_{datetime_str}"
                output_dir_path.mkdir(parents=True, exist_ok=True)

                output_file_path = output_dir_path / f"{id}.csv"

                with open(output_file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["x", "y", "t", "is_confident"])
                    writer.writerows(trajectory.events)

                print(f"Created {output_file_path}!")

    print("Finished!")



                













