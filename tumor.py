# Second attempt pl work 
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imageio
import cv2 

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
MASK_DIR = os.path.join(ROOT_DIR, "train_GT")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import matplotlib.pyplot as plt

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class TumorConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tumor"
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + whole tumor
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    TRAIN_ROIS_PER_IMAGE = 32
    IMAGE_CHANNEL_COUNT = 1
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    # DETECTION_MIN_CONFIDENCE = 0.95
    # DETECTION_NMS_THRESHOLD = 0.0
    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    USE_MINI_MASK= False
    # TRAIN_ROIS_PER_IMAGE = 56
    # MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    DETECTION_MAX_INSTANCES = 1


############################################################
#  Dataset
############################################################

class TumorDataset(utils.Dataset):
    
    def load_tumor(self, dataset_dir, subset):
        """
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # print("start load_tumor")
        # Add classes. We have only one class to add.
        self.add_class("tumor", 1, "tumor")

        # Train or validation dataset?
        print("subset = ", subset)
        assert subset in ["train", "val"]
        # dataset_dir = os.path.join(dataset_dir, subset)
        print("dataset_dir = ", dataset_dir)
        # Store original img and mask 
        image_ids = os.listdir(dataset_dir)
        mask_ids = os.listdir(MASK_DIR)
        # print(mask_ids)
        for i in range(len(image_ids)):
            image_path = os.path.join(dataset_dir, image_ids[i])
            mask_path = os.path.join(MASK_DIR, mask_ids[i])
            # print(mask_path)
            # exit(0)
            image = imageio.imread(image_path)
            # TODO: NEW********************
            image = image[..., np.newaxis]
            mask = imageio.imread(mask_path)
            # print(mask.shape)
            # exit(0)
            mask = mask[..., np.newaxis]
            # print(image.shape)
            # print(mask.shape)
            # exit(0)
            height, width = image.shape[:2]
            self.add_image(
                "tumor",
                image_id=image_ids[i], 
                path=image_path,
                width=width, height=height,
                img_mask=mask
            )
        # print(image_ids)
        # print("end of load_tumor")

    # Returns masks = a boolean array of shape [height, width, instance count]
    # one mask per instance
    # class_ids: 1D arr of class_ids for each mask. (all one for now)
    def load_mask(self, image_id):
        # print("start of load_mask")
        image_info = self.image_info[image_id]
        if image_info["source"] != "tumor":
            return super(self.__class__, self).load_mask(image_id)
        
        info = self.image_info[image_id]
        img_masks = info["img_mask"]
        # print("shape of img_masks", img_masks.shape)
        # print("numner of masks in image: ", len(img_masks))
        shape = [info["height"], info["width"]]
        #exit(1)

        # Mask array placeholder
        mask_array = np.zeros([info["height"], info["width"], len(info["img_mask"])],dtype=np.uint8)
        # for index, mask in enumerate(img_masks):
        #    mask_array[:,:,index] = self.rle_decode(mask, shape)
        return img_masks.astype(np.bool), np.ones([img_masks.shape[-1]], dtype=np.int32)

    def rle_decode(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        '''
        if not isinstance(mask_rle, str):
            img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
            return img.reshape(shape).T

        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tumor":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)   


def train(model):
    """Train the model."""
    # Training dataset.
    # print("entering train")
    dataset_train = TumorDataset()
    dataset_train.load_tumor(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TumorDataset()
    dataset_val.load_tumor(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # ImageNet trained weights, we don't need to train too long. 
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')
    print("Training all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='all')

def detect(model, subset):
    # print(subset)
    # exit(0)
    submit_dir = './results'
    results = []
    image_ids = os.listdir(subset)
    # print(mask_ids)
    for i in range(len(image_ids)):
        image_path = os.path.join(subset, image_ids[i])
        image = imageio.imread(image_path)
        # TODO: NEW********************
        image = image[..., np.newaxis]
        r = model.detect([image], verbose=0)
        results.extend(r)
        visualize.display_instances(
            image, r[0]['rois'], r[0]['masks'], r[0]['class_ids'],
            # image, r[0], r[3], r[1],
            # dataset.class_names, r['scores'],
           ["BG","tumor"], r[0]['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions", figsize=(4,4))
        plt.savefig("{}/{}.png".format(submit_dir, image_ids[i]))

    # print(results)



############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TumorConfig()
    else:
        class InferenceConfig(TumorConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        print("creating model")
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

# Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask", "conv1"])
    else:
        model.load_weights(weights_path, by_name=True,exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask", "conv1"])

    # Train or evaluate
    if args.command == "train":
        train(model)
        # detect(model, args.subset)
    # TODO: NOT SURE IF THIS IS CORRECT
    elif args.command == "detect":
        detect(model, args.subset)
        # print(results)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))