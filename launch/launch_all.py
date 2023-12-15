from signal import SIGINT
from subprocess import Popen
from time import sleep
import os
import psutil

BASE = ["ros2", "launch",
        os.path.dirname(os.path.realpath(__file__)) + "/real_bag.launch.py"]
VARIATIONS = [
    ["which:=0"],
    ["which:=1", "mode:=0"],
    ["which:=1", "mode:=1"],
    ["which:=1", "mode:=2"]
]
DURATION = 10.0

for i in range(4):
    process = Popen(BASE + VARIATIONS[i], text=True)
    sleep(DURATION)

    process.send_signal(SIGINT)
    return_code = process.wait(timeout=10)
    print(f"return_code: {return_code}")

    for proc in psutil.process_iter():
        if "republish" in proc.name():
            proc.kill()

    sleep(3.0)

    root_dir = os.path.dirname(os.path.dirname(
        os.path.realpath(__file__)))
    Popen(["python3", root_dir + "/notebooks/assess_perf.py", str(i)])

    exit()  # stop on first for now
