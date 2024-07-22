
import importlib
from pathlib import Path
import importlib.util
import sys
import os

script_path = "G:/Program Files/Prophesee/share/metavision/sdk/core/python_samples/metavision_file_to_csv/metavision_file_to_csv2.py"

spec = importlib.util.spec_from_file_location("module.name", script_path)
foo = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = foo
spec.loader.exec_module(foo)



raw_path = "D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/mb-bum2-2.raw"
output_dir = "D:/Bibliothek/Workspace/_Studium/MA/aufnahmen/exported_csv/"

foo.main(raw_args=["-i", raw_path, "-o", output_dir])

