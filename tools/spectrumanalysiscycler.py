# files
import pathlib as pl
import os
from io import open
from numpy.lib.function_base import iterable

# manipulation and tabulation
import pandas as pd
import numpy as np

# images
from PIL import Image, ImageFilter, ImageEnhance

# time
from time import time_ns, process_time_ns, sleep

# concurrency
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

# prebuilt utilities
import utilities

def  GetLocus(x: int, y: int, xmax: int, ymax: int): 
    xbuf = (len(str(xmax)) - len(str(x))) * "0" + str(x)
    ybuf = (len(str(ymax)) - len(str(y))) * "0" + str(y)
    return "X" + xbuf + "Y" + ybuf

def ColorMeshCompensate(target: pl.Path):
    # support to follow
    return Image.open(target)

def ImageProcess(target: pl.Path, saveloc: pl.Path, compensate: bool = False):
    LocalTable = pd.DataFrame(columns=["locus", "hex", "transparency",
                                       "red", "green", "blue", 
                                       "white", "whitepercent",
                                       "black", "blackpercent"])
    LocalTable = LocalTable.astype({"locus": "object", "hex": "object", "transparency": "int64",
                                    "red": "int64", "green": "int64", "blue": "int64", 
                                    "white": "int64", "whitepercent": "int64",
                                    "black": "int64", "blackpercent": "int64"})
    start = process_time_ns()
    internal = Image.open(target) if compensate == False else ColorMeshCompensate(target)
    height, width = internal.size
    internal = np.array(internal, np.uint32)
    for i in range(width):
        for j in range(height):
            data = internal[i, j]
            result = utilities.PixelSummary(data)
            result.update({"locus": GetLocus(i, j, width, height)})
            LocalTable = LocalTable.append(result, ignore_index=True)
    total_time = (process_time_ns() - start) / 1000000000
    totalpx = height * width
    fname = str(target.name).rstrip(pl.Path(target.name).suffix)
    LocalTable.to_csv(str(saveloc) + "\\" + fname + ".csv", index=False)
    unique_colors = LocalTable.hex.unique()
    fbuf = open(str(saveloc) + "\\" + fname + "_summary.txt", "w+", encoding="utf-8")
    fbuf.writelines([
        "File: " + str(target) + "\n",
        "Height: " + str(height) + "px\n",
        "Width: " + str(width) + "px\n",
        "Total Pixels: " + str(totalpx) + "px\n"
        "Total Process Time: " + str(total_time) + " seconds\n"
        "Unique Color Count: " + str(len(unique_colors)) + " colors\n"
    ])
    for i in unique_colors:
        count = LocalTable[LocalTable.hex == i].shape[0]
        fbuf.write("\t> " + i + " [" + str(count) + " instances]\n")
    fbuf.writelines([
        "Red Pixel Intensity Base: " + str(LocalTable["red"].min()) + "\n",
        "Red Pixel Intensity Mean: " + str(LocalTable["red"].mean()) + "\n",
        "Red Pixel Intensity Peak: " + str(LocalTable["red"].max()) + "\n",
        "Green Pixel Intensity Base: " + str(LocalTable["green"].min()) + "\n",
        "Green Pixel Intensity Mean: " + str(LocalTable["green"].mean()) + "\n",
        "Green Pixel Intensity Peak: " + str(LocalTable["green"].max()) + "\n",
        "Blue Pixel Intensity Base: " + str(LocalTable["blue"].min()) + "\n",
        "Blue Pixel Intensity Mean: " + str(LocalTable["blue"].mean()) + "\n",
        "Blue Pixel Intensity Peak: " + str(LocalTable["blue"].max()) + "\n"
    ])
    fbuf.close()

# CLI in-line argument support coming soon

def dialog():
    # CLI dialog when no in-line arguments are given 
    bar = "===== ===== ===== ===== ====="
    internal = {}
    print("Color Spectrum Analysis Tool")
    print(bar)
    while(True):
        target = input("Input target file/folder: ")
        if pl.Path(target).exists():
            if pl.Path(target).is_file():
                internal["targets"] = [target]
            else:
                targets = os.listdir(target)
                buf = []
                for i in targets:
                    if pl.Path(target + "/" + i).is_file():
                        buf.append(target + "/" + i)
                internal["targets"] = buf
            break
        else:
            print("Path does not exist. Try again.")
            print(bar)
        del target
    print(bar)
    while(True):
        target = input("Input result save location: ")
        if pl.Path(target).exists():
            if pl.Path(target).is_dir():
                internal["savelocation"] = pl.Path(target)
                break
            else:
                print("Path is not a directory. Try again.")
                print(bar)
        else:
            pl.Path(target).mkdir()
            internal["savelocation"] = pl.Path(target)
            print("Directory creation successful.")
            break
        del target
    print(bar)
    while(True):
        print("Available CPU Cores: " + str(cpu_count() - 2))
        target = int(input("Input CPU Core usage [multiprocessing]: ")) or 0
        if 0 < target <= (cpu_count() - 2):
            internal["workers"] = target
            break
        else:
            print("Usage is out of range. Try Again.")
            print(bar)
        del target
    print(bar)
    return internal

def main(
    targets: iterable,
    savelocation: pl.Path,
    workers: int
):
    ProcessExecutor = ThreadPoolExecutor(max_workers=workers)
    fut = {ProcessExecutor.submit(ImageProcess, pl.Path(target), pl.Path(savelocation)): target for target in targets}
    start = time_ns()
    active = len(ProcessExecutor._threads)
    pending = ProcessExecutor._work_queue.qsize() - workers if ProcessExecutor._work_queue.qsize() - workers > 0 else 0
    while (active > 0 or pending > 0):
        print("Running... Active: {} | Pending: {} | Workers: {} | Elapsed: {:.2f}".format(active, pending, workers, (time_ns() - start) / 1000000000), end="\r")
        active = len(ProcessExecutor._threads)
        pending = ProcessExecutor._work_queue.qsize() - workers if ProcessExecutor._work_queue.qsize() - workers > 0 else 0
        sleep(0.5)
        print(" " * 75, end="\r")
    for i in as_completed(fut):
        print(i)
    ProcessExecutor.shutdown(True, False)

if __name__ == "__main__":
    args = dialog()
    main(**args)
