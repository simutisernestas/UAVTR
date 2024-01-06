# %%
import matplotlib.pyplot as plt
import os
import psutil as psu
import time
import pickle as pkl


def find_procs_by_name(name):
    "Return a list of processes matching 'name'."
    ls = []
    for p in psu.process_iter(['name']):
        if name in p.info['name']:
            ls.append(p)
    return ls


def print_all_procs():
    for proc in psu.process_iter(['pid', 'name']):
        print(proc.info)


print("CPU Percent: ", psu.cpu_percent())
print("CPU Count: ", psu.cpu_count())
TOTAL_MEM = psu.virtual_memory().total / 1024**3
print(f"Total Memory: {TOTAL_MEM} GB")
print_all_procs()

# %%

procs = {}
for name in [
    'orientation_filter',
    'estimation_node',
    'tracking_ros_node'
]:
    proc = find_procs_by_name(name)
    assert len(proc) == 1, f"{name} has {len(proc)} processes"
    proc = proc[0]
    procs[name] = {
        'proc': proc,
        'cpup': [],
        'memp': []
    }


def dump(data_dict, tmstr):
    dump_dict = {}
    for k, v in data_dict.items():
        for k2, v2 in v.items():
            if k2 == 'proc':
                continue
            dump_dict[f"{k}_{k2}"] = v2
    pkl.dump(dump_dict,
             open(f"./data/{tmstr}_proc_stats.pkl", "wb"))


verbose = True
stop = False
timestamp = time.time()
while 1:
    for proc_name, proc_data in procs.items():
        proc = proc_data['proc']
        try:
            cpu_usage = proc.cpu_percent(interval=.5)
            mem_usage = [proc.memory_percent() for _ in range(5)]
        except Exception as e:
            print(f"{proc_name} is dead; err {e}")
            stop = True
            break
        if verbose:
            print(f"Process: {proc_name}")
            print(f"CPU Percent: {cpu_usage}")
            print(f"Memory Percent: {mem_usage}")

        procs[proc_name]['cpup'].append(cpu_usage)
        procs[proc_name]['memp'].append(mem_usage)

    dump(procs, int(timestamp))

    if stop:
        break

# %%

pickles = [x for x in os.listdir("./data") if "proc_stats" in x]
pickles.sort()
latest = pickles[-1]
data = pkl.load(open(f"./data/{latest}", "rb"))

# element wise sum of all all three nodes cpup
summed_cpup = [sum(x) for x in zip(
    data["orientation_filter_cpup"],
    data["estimation_node_cpup"],
    data["tracking_ros_node_cpup"]
)]

# merge list of lists into one list
data["orientation_filter_memp"] = [
    item for sublist in data["orientation_filter_memp"] for item in sublist]
data["estimation_node_memp"] = [
    item for sublist in data["estimation_node_memp"] for item in sublist]
data["tracking_ros_node_memp"] = [
    item for sublist in data["tracking_ros_node_memp"] for item in sublist]

summed_memp = [sum(x)/100 * TOTAL_MEM for x in zip(
    data["orientation_filter_memp"],
    data["estimation_node_memp"],
    data["tracking_ros_node_memp"]
)]

plt.figure(figsize=(7, 5), dpi=200)
plt.hist(summed_cpup, bins='auto', rwidth=.8, label="CPU Usage",
         color='green', range=(300, 480))
plt.xlabel("CPU [%]")
plt.ylabel("Samples")
plt.legend()
plt.tight_layout()
plt.savefig("./plots/cpu_usage.png")
plt.show()
plt.figure(figsize=(7, 5), dpi=200)
plt.hist(summed_memp, bins='auto', rwidth=.8, range=(.865, .905),
         color='green', label="Memory Usage")
plt.xlabel("Memory [GB]")
plt.ylabel("Samples")
plt.tight_layout()
plt.legend()
plt.savefig("./plots/memory_usage.png")
plt.show()

# %%

# read in data from data/estimation_node_22074_1704467489560.log
with open("./data/estimation_node_14990_1704552700774.log", "r") as f:
    est_lines = f.readlines()
with open("./data/orientation_filter_14874_1704552697959.log", "r") as f:
    orien_lines = f.readlines()

times_est = [float(x.split(" ")[-2])
             for x in est_lines if "imu callback took" in x]
times_ort = [float(x.split(" ")[-2])
             for x in orien_lines if "imu callback took" in x]

times = [sum(x) for x in zip(times_est, times_ort[:len(times_est)])]
times = [x for x in times if x > .3]

plt.figure(figsize=(7, 5), dpi=200)
plt.hist(times, bins='auto', rwidth=.8,
         alpha=.9, label="IMU Fusion Latency",
         color='green')
plt.xlabel("Latency [ms]")
plt.ylabel("Samples")
plt.legend()
plt.tight_layout()
plt.savefig("./plots/state_latency.png")
plt.show()
