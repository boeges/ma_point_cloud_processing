
import importlib
from pathlib import Path
import importlib.util
import sys
import os


def import_module(script_path):
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    return module

def raw_to_csv(module, raw_path, output_dir, startts, endts):
    args = ["-i", raw_path, "-o", output_dir]
    if startts > 0:
        args.append("-s")
        args.append(str(startts))
    args.append("-d")
    if startts > 0:
        args.append(str(endts))
    else:
        args.append(str(60*60*1_000_000)) # max 1h

    print("Converting", raw_path, "using args:", args)
    module.main(raw_args=args)


if __name__ == "__main__":
    script_path = "G:/Program Files/Prophesee/share/metavision/sdk/core/python_samples/metavision_file_to_csv/metavision_file_to_csv2.py"
    output_dir = "D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-08-06_libellen/exported_csv/"

    module = import_module(script_path)

    # format: path, start ts in us, end ts in us
    # raw_paths = [
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee1-1.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee1-2.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee1-3.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee1-4.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee1-5.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee1-6.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee1-7.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee1-8.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee1-9.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee1-10.raw", 0, -1),

    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-1.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-2.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-3.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-4.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-5.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-6.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-7.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-8.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-9.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-10.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bee2-11.raw", 0, -1),
        
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum1-1.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum1-2.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum1-3.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum1-4.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum1-5.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum1-6.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum1-7.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum1-8.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum1-9.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum1-10.raw", 0, -1),

    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum2-1.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum2-2.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum2-3.raw", 0, -1),
    #     ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-07-18_bunter_garten/mb-bum2-4.raw", 0, -1),
    # ]

    raw_paths = [
        ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-08-06_libellen/mb-dra1-1.raw", 0, -1),
        ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-08-06_libellen/mb-dra1-2.raw", 0, -1),
        ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-08-06_libellen/mb-dra1-3.raw", 0, -1),
        ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-08-06_libellen/mb-dra1-4.raw", 0, -1),
        ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-08-06_libellen/mb-dra2-1.raw", 0, -1),
        ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-08-06_libellen/mb-dra2-2.raw", 0, -1),
        ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-08-06_libellen/mb-dra2-3.raw", 0, -1),
        ("D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/2024-08-06_libellen/mb-dra2-4.raw", 0, -1),
    ]

    for path,sts,ets in raw_paths:
        raw_to_csv(module, path, output_dir, sts, ets)



