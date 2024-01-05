# %%
import psutil as psu
import time


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
# print_all_procs()

# %%

proc = find_procs_by_name('firefox')
assert len(proc) == 1, len(proc)
proc = proc[0]
print(proc.info)

for _ in range(5):
    print(f"CPU Percent: {proc.cpu_percent(interval=1)}")
    print(f"Memory Percent: {proc.memory_percent()}")
    # time.sleep(.2)
