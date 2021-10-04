# files
import pathlib as pl
import os
from io import open
from numpy.lib.function_base import iterable

# manipulation and tabulation
import pandas as pd
import numpy as np

# images
from PIL import Image

# time
from time import time_ns, process_time_ns, sleep

# concurrency
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from utilities.colordata import PixelSummary

# prebuilt utilities
import utilities

bar = "===== ===== ===== ===== ====="

def GetID(x, xmax):
    return (len(str(xmax)) - len(str(x))) * "_" + str(x)

def  GetLocus(x: int, y: int, xmax: int, ymax: int): 
    xbuf = (len(str(xmax)) - len(str(x))) * "0" + str(x)
    ybuf = (len(str(ymax)) - len(str(y))) * "0" + str(y)
    return "X" + xbuf + "Y" + ybuf

def ColorMeshCompensate(target: pl.Path):
    # support to follow
    return Image.open(target)

def processchunk(
    section: int,
    chunk: np.ndarray,
    dodgewhite: bool,
    bufdat: tuple
):
    buffer = []
    for j in range(len(chunk)):
        data = chunk[j]
        result = PixelSummary(GetLocus(section, j, bufdat[0], bufdat[1]), data)
        if dodgewhite == True and result["hex"] == "#FFFFFF":
            continue
        else:
            buffer.append(result)
    return buffer

def ImageProcess(
    target: pl.Path, 
    saveloc: pl.Path,
    subworkers: int,
    dodgewhite: bool,
    compensate: bool = False
):
    LocalTable = pd.DataFrame(columns=["locus", "hex", "transparency",
                                       "red", "green", "blue", 
                                       "white", "whitepercent",
                                       "black", "blackpercent"])
    LocalTable = LocalTable.astype({"locus": "object", "hex": "object", "transparency": "int64",
                                    "red": "int64", "green": "int64", "blue": "int64", 
                                    "white": "int64", "whitepercent": "int64",
                                    "black": "int64", "blackpercent": "int64"})
    LocalExecutor = ThreadPoolExecutor(max_workers=subworkers)
    start = process_time_ns()
    internal = Image.open(target) if compensate == False else ColorMeshCompensate(target)
    width, height = internal.size
    internal = np.array(internal, np.uint32)
    fut = {LocalExecutor.submit(processchunk, 
                                i, 
                                internal[:, i, :],
                                dodgewhite,
                                (width, height))
           : i for i in range(width)
           }
    LocalExecutor.shutdown(wait=True, cancel_futures=False)
    for buf in as_completed(fut):
        result = buf.result()
        LocalTable = LocalTable.append(buf.result(), ignore_index=True, sort=False)
    total_time = (process_time_ns() - start) / 1000000000
    LocalTable = LocalTable.sort_values(by="locus")
    totalpx = height * width
    fname = str(target.name).rstrip(pl.Path(target.name).suffix)
    LocalTable.to_csv(str(saveloc) + "\\" + fname + ".csv", index=False)
    unique_colors = dict(LocalTable.hex.value_counts())
    fbuf = open(str(saveloc) + "\\" + fname + "_summary.txt", "w+", encoding="utf-8")
    fbuf.writelines([
        bar + "\n"
        "File: " + str(target) + "\n",
        "Height: " + str(height) + "px\n",
        "Width: " + str(width) + "px\n",
        bar + "\n",
        "Total Pixels: " + str(totalpx) + "px\n"
        "Total Process Time: " + str(total_time) + " seconds\n",
        bar + "\n",
        "Red Pixel Color Intensity Statistics" + "\n",
        "\tBase: " + str(LocalTable["red"].min()) + "\n",
        "\tMean: " + str(LocalTable["red"].mean()) + "\n",
        "\tPeak: " + str(LocalTable["red"].max()) + "\n",
        bar + "\n",
        "Green Pixel Color Intensity Statistics" + "\n",
        "\tBase: " + str(LocalTable["green"].min()) + "\n",
        "\tMean: " + str(LocalTable["green"].mean()) + "\n",
        "\tPeak: " + str(LocalTable["green"].max()) + "\n",
        bar + "\n",
        "Blue Pixel Color Intensity Statistics" + "\n",
        "\tBase: " + str(LocalTable["blue"].min()) + "\n",
        "\tMean: " + str(LocalTable["blue"].mean()) + "\n",
        "\tPeak: " + str(LocalTable["blue"].max()) + "\n",
        bar + "\n",
        "Unique Color Count: " + str(len(unique_colors)) + " colors\n"
    ])
    index = 1
    for key, value in unique_colors.items():
        fbuf.write("\t" + GetID(index, len(unique_colors)) + ": " + key + " [" + str(value) + " instances]\n")
        index += 1
    fbuf.close()

# CLI in-line argument support coming soon

def dialog():
    # CLI dialog when no in-line arguments are given 
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
        target = int(input("Input simultaneous subprocesses [multiprocessing]: ")) or 0
        if 0 < target <= (cpu_count() - 2):
            internal["workers"] = target
            break
        else:
            print("Usage is out of range. Try Again.")
            print(bar)
        del target
    print(bar)
    while(True):
        availthr = (cpu_count() - 2)
        usablethr = (availthr - (availthr % internal["workers"])) / internal["workers"]
        print("Available Threads per Core: " + str(usablethr))
        target = int(input("Input simultaneous threads [multithreading]: ")) or 0
        if 0 < target <= usablethr:
            internal["subworkers"] = target
            break
        else:
            print("Usage is out of range. Try Again.")
            print(bar)
        del target
    print(bar)
    while(True):
        target = input("Dodge white [#FFFFFF] pixels (y/n): ").lower()
        if target in ["y", "n"]:
            if target == "y":
                print("White pixels will dodged.")
                internal["dodgewhite"] = True
                break
            elif target == "n":
                print("White pixels will not be dodged.")
                internal["dodgewhite"] = False
                break
        else:
            print("Unknown input. Try Again.")
            print(bar)
        del target
    print(bar)
    return internal

def main(
    targets: iterable,
    savelocation: pl.Path,
    subworkers: int,
    dodgewhite: bool,
    workers: int
):
    ProcessExecutor = ProcessPoolExecutor(max_workers=workers)
    fut = {ProcessExecutor.submit(ImageProcess, 
                                  pl.Path(target), 
                                  pl.Path(savelocation),
                                  subworkers,
                                  dodgewhite): target for target in targets}
    start = time_ns()
    completed = 0
    while (completed != len(targets)):
        active = 0
        rusk = 0
        for i in fut:
            if "state=running" in str(i):
                active += 1
            if "state=finished" in str(i):
                rusk += 1
        completed = rusk
        pending = len(targets) - (completed + active)
        print("Running... Active: {} | Pending: {} | Completed: {} | Elapsed: {:.2f}".format(active, pending, completed, (time_ns() - start) / 1000000000), end="\r")
        sleep(0.5)
        print(" " * 75, end="\r")
    for i in as_completed(fut):
        print(i)
    print("Cycle Complete!")

if __name__ == "__main__":
    args = dialog()
    main(**args)
