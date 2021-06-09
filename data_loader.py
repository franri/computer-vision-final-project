import numpy as np
from glob import glob
import cv2 
from feature_extraction import FeatureExtractors
from feature_extraction import extract_features

def load_training_data(training_positive_dir,trainign_negative_dir, feature_extractor=FeatureExtractors.MiniImage):
    ''' Function for loading loading training data from positive and negative examples
    '''
    positive_img_files = sorted(glob(training_positive_dir + '/*'))
    negative_img_files = sorted(glob(trainign_negative_dir + '/*'))
    #comment this line for loading all data
    positive_img_files = positive_img_files[:100]
    negative_img_files = negative_img_files[:200]

    training_data = []
    training_labels = []
    
    print('##Loading {} positive face images'.format(len(positive_img_files)))
    for img in positive_img_files:
        image = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
        image_representation = extract_features(feature_extractor,image)
        training_data.append(image_representation)
        training_labels.append(1)
    
    print('##Loading {} negative face images'.format(len(negative_img_files)))
    for img in negative_img_files:
        image = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
        image_representation = extract_features(feature_extractor,image)
        training_data.append(image_representation)
        training_labels.append(0)   
    
    training_data = np.asarray(training_data)
    training_labels = np.asarray(training_labels)
    return training_data, training_labels

def load_validation_data(validation_data_dir):

    validation_image_files = sorted(glob(validation_data_dir + '/*'))
    val_images = []
    for img_file in validation_image_files:
        image = cv2.imread(img_file,cv2.IMREAD_COLOR)
        val_images.append(image)

    return val_images 
   