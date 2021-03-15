import os
import glob
import sys

files = glob.glob("%s/*_color.mp4" %sys.argv[1])
print(files)
print(len(files))

for file in files:
    path, name = os.path.split(file)
    print(path, name)
    pref, suff = name.split('.')
    dir_name1 = sys.argv[2]
    if not os.path.isdir(dir_name1):
        os.mkdir(dir_name1)
    dir_name2 = "/%s" %(pref)
    dir_name = dir_name1+dir_name2
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    os.system("./build/examples/openpose/openpose.bin --video %s --hand --face -frame_step 2 -write_json %s -display 0 -render_pose 0" %(file, dir_name))

