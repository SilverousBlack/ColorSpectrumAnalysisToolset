from core import *

bar = "===== ===== ===== ===== ====="

GlobalProcessPool = ProcessPoolExecutor()

def cls():
    if os.name == "nt":
        os.system("cls")
    elif os.name == "posix":
        os.system("clear")

def GetID(x, xmax):
    return (len(str(xmax)) - len(str(x))) * "_" + str(x)

def  GetLocus(x: int, y: int, xmax: int, ymax: int): 
    xbuf = (len(str(xmax)) - len(str(x))) * "0" + str(x)
    ybuf = (len(str(ymax)) - len(str(y))) * "0" + str(y)
    return "X" + xbuf + "Y" + ybuf

def ColorMeshCompensate(
    dominant: list,
    strain: tuple,
    tolerance: float = 1.5,
):
    if strain in dominant:
        return ("Dominant", utilities.GetHexString(strain), 100.00)
    buffer = {}
    for dom in dominant:
        dom = np.array(dom)
        strain = np.array(strain)
        score = abs((255 - DE_Euclid(strain, dom)) / 256) * 100
        val = utilities.colordata.GetHexString(dom)
        buffer[score] = ("Mesh" if score >= tolerance else "Rare Unique", val, score)
    return buffer[np.max(np.array(list(buffer.keys())))]

def FeedCompensate(
    dominant: list,
    buffer: pd.Series,
    tolerance: int,
):
    state = []
    relative = []
    percent = []
    for i in buffer:
        stat, rel, per = ColorMeshCompensate(dominant, utilities.ColorDataFromHex(i), tolerance)
        state.append(stat)
        relative.append(rel)
        percent.append(per)
    return (pd.Series(state), pd.Series(relative), pd.Series(percent))

def processchunk(
    section: int,
    chunk: np.ndarray,
    dodgewhite: bool,
    bufdat: tuple
):
    buffer = []
    for j in range(len(chunk)):
        data = chunk[j]
        result = utilities.PixelSummary(GetLocus(section, j, bufdat[0], bufdat[1]), data)
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
    compensate: bool = False,
    dominance: float = 1.5,
    tolerance: float = 1.5
):
    try:
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
        internal = Image.open(target)
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
            LocalTable = LocalTable.append(buf.result(), ignore_index=True, sort=False)
        total_time = (process_time_ns() - start) / 1000000000
        LocalTable = LocalTable.sort_values(by="locus")
        totalpx = height * width
        fname = str(target.name).rstrip(pl.Path(target.name).suffix)
        LocalTable.to_csv(str(saveloc) + "\\" + fname + ".csv", index=False)
        unique_colors = dict(LocalTable.hex.value_counts())
        color_data = []
        dominant_colors = []
        for key, value in unique_colors.items():
            percent = value / (height * width) * 100
            color_data.append({"hex": key,
                           "instances": value,
                           "percent": percent})
            if percent >= dominance:
                dominant_colors.append(utilities.ColorDataFromHex(key))
        color_table = pd.DataFrame(data=color_data, columns=["hex", "instances", "percent"])
        if compensate == True:
            color_table["state"], color_table["e-relative"], color_table["e-diff"] = FeedCompensate(dominant_colors, color_table["hex"], tolerance)
        else:
            color_table["state"], color_table["e-relative"], color_table["e-diff"] = FeedCompensate(dominant_colors, color_table["hex"], 0)
        color_table.to_csv(str(saveloc) + "\\" + fname + "_colors.csv", index=False)
        del color_data, dominant_colors
        unique_colors = color_table.loc[color_table["state"].isin(["Dominant", "Rare Unique"])].to_numpy()
        fbuf = open(str(saveloc) + "\\" + fname + "_summary.txt", "w+", encoding="utf-8")
        fbuf.writelines([
            bar + "\n",
            "File: " + str(target) + "\n",
            "Height: " + str(height) + "px\n",
            "Width: " + str(width) + "px\n",
            bar + "\n",
            "Total Pixels: " + str(totalpx) + "px\n"
            "Total Process Time: " + str(total_time) + " seconds\n",
            "Dodging White [#FFFFFF] pixels: " + str(dodgewhite) + "\n",
            "Mass Dominance Rate: " + str(dominance) + "\n",
            "Compensating for Color Mesh: " + str(compensate) + "\n",
            ("Color Mesh Tolerance: " + str(tolerance) + "\n") if compensate == True else "Auto relative color calculation tolerance rate: 0\n",
            "Color Difference Mode: Euclidean\n",
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
        for value in unique_colors:
            cbuf = pd.DataFrame(color_table[color_table["e-relative"].isin([value[0]])])
            mcount = cbuf.shape[0] - 1
            mtotal = cbuf["instances"].sum() - value[1]
            mpercent = cbuf["percent"].sum() - value[2]
            fbuf.writelines(["\t{0}: {1} [{2} instances, {3:.2f}%, {4}]\n".format(
                GetID(index, len(unique_colors)),
                value[0], value[1], value[2], value[3]
            ) if value[2] >= 0.01 else "\t{0}: {1} [{2} instances, {3:.2e}%, {4}]\n".format(
                GetID(index, len(unique_colors)),
                value[0], value[1], value[2], value[3]
            ), 
                "\t\t> Mesh: {0} colors [{1}, {2:.2f}%]\n".format(
                    mcount, 
                    mtotal, 
                    mpercent) if mcount > 1 else ""])
            index += 1
        fbuf.close()
        return "Process Success"
    except Exception as e:
        return "Process hit an error: " + str(e)
    
def main(
    targets: Iterable,
    savelocation: pl.Path,
    subworkers: int,
    dodgewhite: bool,
    workers: int,
    compensate: bool = False,
    dominance: float = 1.5,
    tolerance: float = 15,
):
    ProcessExecutor = ProcessPoolExecutor(max_workers=workers)
    fut = {ProcessExecutor.submit(ImageProcess, 
                                  pl.Path(target), 
                                  pl.Path(savelocation),
                                  subworkers,
                                  dodgewhite,
                                  compensate,
                                  dominance,
                                  tolerance): target for target in targets}
    cls()
    start = time_ns()
    active = workers
    completed = 0
    pending = len(targets) - (completed + active)
    while (completed != len(targets)):
        print(bar)
        print("Targets: ")
        for i in targets:
            print("\t> " + i)
        print("Simultaneous Processes: " + str(workers))
        print("Simultaneous Threads per Process: " + str(subworkers))
        print(bar)
        print("Elapsed: {:.2f} seconds".format((time_ns() - start) / 1000000000))
        print("Active (working): " + str(active) + " jobs")
        print("In Queue (waiting): " + str(pending) + " jobs")
        print("Completed (finished): " + str(completed) + " jobs")
        print(bar)
        active = 0
        completed = 0
        for i in fut:
            buffer = str(i).rstrip(">").split(" ")
            print("{}: {}".format(buffer[2], buffer[3].rstrip()))
            if buffer[3] == "state=running":
                active += 1
            if buffer[3] == "state=finished":
                completed += 1
        pending = len(targets) - (completed + active)
        sleep(0.5)
        cls()
    for i in as_completed(fut):
        buffer = str(i).rstrip(">").split(" ")
        print("{}: {}".format(buffer[2], i.result()))
    print("Cycle Complete!")
    