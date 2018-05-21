import tensorflow as tf
import os
import pdb


def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename = line[:-1].split(" ")[0]
        label = line[:-1].split(" ")[1]
        filenames.append(filename)
        labels.append(str(label))
    return filenames, labels

# Reads pfathes of images together with their labels
filename = "./filename_images"
image_list, label_list = read_labeled_image_list(filename)

images = tf.convert_to_tensor(image_list, dtype=tf.string)
labels = tf.convert_to_tensor(label_list, dtype=tf.string)

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels],
                                            shuffle=True)
