import os
import shutil
import argparse
import re
from itertools import count


GLOBAL_COUNTER = count()


def remove_augmented_files(src):

    for item in os.listdir(src):
        s = os.path.join(src, item)
        filename = os.path.basename(s)
        if "_" in filename:
            try:
                os.remove(s)
                print("Removed file: " + filename)
            except OSError as e:  # if failed, report it back to the user ##
                print("Error: %s" % (filename))

        # split = item.split('_')
        # print(s)
        # print(split)
        # # Handle errors while calling os.remove()
        # try:
        #     os.remove(s)
        # except:
        #     print("Error while deleting file ", s)


def file_counter():
    print("{} of {} files completed.".format(next(GLOBAL_COUNTER)))


def main():
    src=r'C:\Users\A560655\Documents\datasets\bird_polar bear'
    remove_augmented_files(src)


if __name__ == "__main__":
    main()
