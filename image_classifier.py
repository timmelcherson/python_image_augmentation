#%%
import os
import shutil

def navigate_and_rename(src):
    jpgCounter = 0
    txtCounter = 0

    for item in os.listdir(src):

        s = os.path.join(src, item)
        print("item: " + item)
        print(s)
        # split = item.split('.')

        # if s.endswith(".jpg"): 

        #     jpg_filename = str(split[0] + str(jpgCounter) + "." + split[1])
        #     shutil.copy(s, os.path.join(src, jpg_filename))   
        #     jpgCounter += 1
        #     print(jpg_filename)
        # elif s.endswith(".txt"):
        #     txt_filename = split[0] + str(txtCounter) + "." + split[1]
        #     shutil.copy(s, os.path.join(src, txt_filename)) 
        #     txtCounter += 1
        #     print(txt_filename)
        # else:
        #     print("Neither .jpg file or .txt file")

         

directory = 'images2'
navigate_and_rename(directory)

# %%
