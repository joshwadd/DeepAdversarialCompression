import numpy as np
import os
import glob
import shutil
from datetime import datetime
from scipy.ndimage import imread
import math

# the height and width of the full frames to test on.
FULL_HEIGHT = 1080     #was 1024
FULL_WIDTH = 1920
# the height and width of the patches to train on
TRAIN_HEIGHT = TRAIN_WIDTH = 128
#Number of mixture componets
NUMBER_OF_MIXTURE = 6
GSM_SAMPLE = 24576
#Important rate distorition tradeoff parameter
ALPHA = 0.4

DIM_REDUCTION = 10

#Type of quantisation model to use
    #True :   rounding
    #False : iid noise
QUANT_MODEL = True

#If the adversial training process should be balanced as a function
#of the prediction accuracy
BABYSIT = False
LOWER = 0.8
UPPER = 0.95
SMOOTHING =0.8




DIM_REDUCE_SIZE = math.ceil((TRAIN_HEIGHT*TRAIN_WIDTH*3)/(16*16*DIM_REDUCTION))

SCALE_ENTROPY = (1.0/(16*16*DIM_REDUCE_SIZE))
SCALE_RECON =  ((1.0)/(1000.0*128*128*3))

##
# General training
##

# whether to use adversarial training vs. basic training of the generator
ADVERSARIAL = True
# the training minibatch size
BATCH_SIZE = 32
BATCH_SIZE_TEST = 6     #should be 6
##
# Loss parameters
##

# for lp loss. e.g, 1 or 2 for l1 and l2 loss, respectively)
L_NUM = 2
# the power to which each gradient term is raised in GDL loss
ALPHA_NUM = 1
# the percentage of the adversarial loss to use in the combined loss
LAM_ADV = 0.1
# the percentage of the lp loss to use in the combined loss
#LAM_LP = 0.8
# the percentage of the GDL loss to use in the combined loss
LAM_GDL = 1

#
# Generator model
##

# learning rate for the generator model
# Default of 0.0001 works well for the first 100k iterations of CAE
LRATE_G = 0.001  #

##
# Discriminator model
##

# learning rate for the discriminator model
LRATE_D = 0.005



##
# Data
##

def get_date_str():
    """
    @return: A string representing the current date/time that can be used as a directory name.
    """
    return str(datetime.now()).replace(' ', '_').replace(':', '.')[:-10]

def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def clear_dir(directory):
    """
    Removes all files in the given directory.

    @param directory: The path to the directory.
    """
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

def get_test_frame_dims():
    video_path = glob.glob(TEST_DIR + '*')[0]
    img_path = glob.glob(video_path + '/*.png')[0]
    img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]

def get_train_frame_dims():
    #img_path = glob(os.path.join(TRAIN_DIR, '*/*'))[0]
    video_path = glob.glob(TRAIN_DIR + '*')[0]
    #img = imread(img_path, mode='RGB')
    img_path =  glob.glob(video_path + '/*.png')[0]
    img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]

def set_test_dir(directory):
    """
    Edits all constants dependent on TEST_DIR.

    @param directory: The new test directory.
    """
    global TEST_DIR, FULL_HEIGHT, FULL_WIDTH

    TEST_DIR = directory
    FULL_HEIGHT, FULL_WIDTH = get_test_frame_dims()

# root directory for all data
DATA_DIR = get_dir('../Data/')
# directory of unprocessed training frames
#TRAIN_DIR = os.path.join(DATA_DIR, 'Ms_Pacman/Train/')
TRAIN_DIR = os.path.join('/media/josh/DATADRIVE1/SurveillanceVideoData/Train/')
# directory of unprocessed test frames
#TEST_DIR = os.path.join(DATA_DIR, 'Test/')
TEST_DIR = DATA_DIR + 'Test/'
# Directory of processed training clips.
# hidden so finder doesn't freeze w/ so many files. DON'T USE `ls` COMMAND ON THIS DIR!
TRAIN_DIR_CLIPS = get_dir(os.path.join(DATA_DIR, '.Images_alt/'))

# For processing clips. l2 diff between frames must be greater than this
MOVEMENT_THRESHOLD = 0
# total number of processed clips in TRAIN_DIR_CLIPS
NUM_CLIPS = len(glob.glob(TRAIN_DIR_CLIPS + '*'))



##
# Output
##

def set_save_name(name):
    """
    Edits all constants dependent on SAVE_NAME.

    @param name: The new save name.
    """
    global SAVE_NAME, MODEL_SAVE_DIR, SUMMARY_SAVE_DIR, IMG_SAVE_DIR

    SAVE_NAME = name
    MODEL_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Models/', SAVE_NAME))
    SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Summaries/', SAVE_NAME))
    IMG_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Images/', SAVE_NAME))

def clear_save_name():
    """
    Clears all saved content for SAVE_NAME.
    """
    clear_dir(MODEL_SAVE_DIR)
    clear_dir(SUMMARY_SAVE_DIR)
    clear_dir(IMG_SAVE_DIR)


# root directory for all saved content
SAVE_DIR = get_dir('../Save/')

# inner directory to differentiate between runs
SAVE_NAME = 'DAC_round_'+ str(DIM_REDUCTION) +'_' + str(ALPHA) + '/'

# directory for saved models
#MODEL_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Models/', SAVE_NAME))
MODEL_SAVE_DIR = get_dir(SAVE_DIR + 'Models/'+ SAVE_NAME)

# directory for saved TensorBoard summaries
#SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Summaries/', SAVE_NAME))
SUMMARY_SAVE_DIR = get_dir(SAVE_DIR + 'Summaries/' + SAVE_NAME)
# directory for saved images
IMG_SAVE_DIR = get_dir(SAVE_DIR + 'Images/' + SAVE_NAME)


STATS_FREQ      = 10     # how often to print loss/train error stats, in # steps
SUMMARY_FREQ    = 10    # how often to save the summaries, in # steps
IMG_SAVE_FREQ   = 1000   # how often to save generated images, in # steps
TEST_FREQ       = 10000   # how often to test the model on test data, in # steps
MODEL_SAVE_FREQ = 20000  # how often to save the model, in # steps

