import os
import shutil
import argparse
import re
from itertools import count


GLOBAL_COUNTER = count()

# A function to remove all the augmented variants of images and keep only the original
# Based on augmented images being named correctly, i.e containing an _ after the original's filename
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

def file_counter():
    print("{} of {} files completed.".format(next(GLOBAL_COUNTER)))


def main():
  
    script_dir = os.path.dirname(__file__)
    src = os.path.join(script_dir, './augmented_test_images/')
    
    remove_augmented_files(src)


if __name__ == "__main__":
    main()
