import os
import pdb

file  = open("filename_images", "w")

for root, dirs, files in os.walk('./train'):
    for name in files:
        file_path = os.path.join(root, name)
        label = root[-2:]
        line = file_path + " " + label + "\n"
        file.write(line)
