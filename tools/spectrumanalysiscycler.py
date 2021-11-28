from SpectrumAnalysisTools import *

# CLI in-line argument support coming soon

def dialog():
    # CLI dialog when no in-line arguments are given 
    cls()
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
                        buf.append(pl.Path(target + "/" + i).as_posix())
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
    while(True):
        target = float(input("Input dominance rate (recommended: 1.5): ")) or 0
        if 0 < target:
            internal["dominance"] = target
            break
        else:
            print("Cannot be zero or negative. Try Again.")
            print(bar)
        del target
    print(bar)
    while(True):
        target = input("Compensate for color mesh (y/n): ").lower()
        if target in ["y", "n"]:
            if target == "y":
                print("Will compensate for color mesh.")
                internal["compensate"] = True
                break
            elif target == "n":
                print("Will not compensate for color mesh.")
                internal["compensate"] = False
                break
        else:
            print("Unknown input. Try Again.")
            print(bar)
        del target
    print(bar)
    while(internal["compensate"]):
        target = float(input("Input compensation tolerance rate (recommended: 1.5): ")) or 0
        if 0 < target:
            internal["tolerance"] = target
            break
        else:
            print("Cannot be zero or negative. Try Again.")
            print(bar)
        del target
    return internal

if __name__ == "__main__":
#    ImageProcess(
#        pl.Path("G:\\Git\\ColorSpectrumAnalysisToolset\\docs\\test images\\small.png"),
#        pl.Path("G:\\Git\\ColorSpectrumAnalysisToolset\\docs\\csa"),
#        6, False, False, 1.5, 1.5
#    )
    args = dialog()
    main(**args)
