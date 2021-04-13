# for degradation experiment in directory take the first csv file with resnet in title
# that is, for degradation in exp in directory that contains either the term contrast, lowpass, highpass, or noise
import os
import re
from csv import DictReader
from image_manipulation import load_image, contrast, noise, highpass, lowpass, save_img
# import image_manipulation
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter
from scipy import fftpack as fp
from skimage.io import imread, imsave
# from scipy.misc import toimage
from PIL import Image
import numpy as np
import sys

experiments = ['contrast', 'noise', 'highpass', 'lowpass']
experiments = [word + "-experiment" for word in experiments]
path = "../raw-data/TF"

def find_treated_images():
    """Finds the images that have been used by resnet for predictions.
    Only session 1 is used respectively. The function returns a list of the images.
    Each element is itself a list with: 
    - the complete image name with condition, 
    - object category (ground truth), 
    - experiment type,
    - experiment condition (level),
    - object response
    - raw image name (as used in ImageNet)
    """
    treated_images = []
    for directory in os.listdir(path):
        if directory in experiments:
            print(directory)
            # take the first csv file with resnet in title
            csv_files = os.listdir(os.path.join(path, directory))
            for csv_file in csv_files:
                if "resnet" in csv_file and "session_1" in csv_file:
                    file = os.path.join(path, directory, csv_file)
                    with open(file, 'r') as read_obj:
                        csv_dict_reader = DictReader(read_obj)
                        # for line in csv file get imagename, condition, experiment, and response
                        for row in csv_dict_reader:
                            img = row['imagename'].split("_")[-2:]
                            img = "_".join(img)
                            lst = [row['imagename'], row['category'], directory.replace("-experiment", ""), row['condition'], row['object_response'], img]
                            treated_images.append(lst)
    return treated_images

treated_images = find_treated_images()             

print(treated_images[0:6])
print(len(treated_images)) # 5120

def existence_treated_images():
    """ checks whether the treated images exist in our library
    takes as input the raw image name (e.g. n03041632_43625.JPEG)
    """
    pass

if not os.path.exists("treated_images/"):
    os.makedirs("treated_images/")
print("dir made")

#apply image degradation:
counter = 0
dc_truth_response = dict()
out_dir = "treated_images/"
for image in treated_images:
    # if image does not already exist in target directory do the following #TODO
    # print(image)
    raw_img_name = image[-1]
    # if raw img name end with file ending .png and cannot be found FileNotFoundError,
    # turn it to jpeg and try again
    # print(raw_img_name)
    img = load_image(raw_img_name)
    try:
        if image[2] == "contrast":
            lvl = float(image[3].strip("c")) * 0.01
            img = contrast(img, lvl)
            image[3]
            exp = "con"

        if image[2] == "noise":
            lvl = float(image[3].strip("nse"))
            img = noise(img, lvl)
            exp = "nse"

        if image[2] == "highpass":
            lvl = image[3].strip("hp")
            if lvl == "inf":
                lvl = 999
            else:
                lvl = float(lvl)
            img = highpass(img, lvl)
            exp = "hp"

        if image[2] == "lowpass":
            lvl = float(image[3].strip("lp"))
            img = lowpass(img, lvl)
            exp = "lp"
        ground_truth = image[1]
        prediction = image[-2]
        
        image_file = f"{counter}_{exp}_dnn_{image[3]}_gt{ground_truth}_pred{prediction}_{raw_img_name}"
        dc_truth_response[f"{raw_img_name}"] = (ground_truth, prediction, ground_truth==prediction)
        # print(image_file)
        # imgdata = np.asarray(img)
        save_img(img, out_dir + image_file, use_JPEG=False)
        # print("processed image number", counter)
        counter += 1

    except:
        raise RuntimeError("image could not be degraded, skipping image.")

print(counter)

# in dict, count no of ground truth categories
distr = dict()
for i in dc_truth_response.values():
    if not i[0] in distr:
        distr[f"{i[0]}"] = 1
    else:
        distr[f"{i[0]}"] += 1

for key, value in distr.items():
    print(f"{key}: {value} occurences")

# iterate through dict and count how often there is agreement btw ground truth and prediction

# output table to see agreement for each category