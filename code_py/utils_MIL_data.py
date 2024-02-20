#######################################################################################
############ MIL DATASET AND DATA LOADER FOR ITERATION THROUGH MINIBATCHES ############
#######################################################################################

import os
import random
import collections
import cv2
import csv
import pickle
import numpy as np
import pandas as pd
import time
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

random.seed(42)
torch.manual_seed(0)
np.random.seed(0)

############################################# MIL DATASET ##################################################
class MILDataset(torch.utils.data.Dataset):

    def __init__(self, data_frame):

        """Dataset object for MIL embedings.
            Dataset object which aims to read data from a dataframe and load the patch-level embeddings
        Args:
          data_frame: pandas dataframe with ground truth information for each bag.
        Returns:
          MILDatasetObject object
        Past Updates: Pablo M. (01/2024)
        """

        self.data_frame = data_frame
        self.classes = "GT"
        self.images = self.data_frame['Patient'].values # WSI identifiers
        self.targets = self.data_frame['GT'].values # Ground truth

        # Adaptation for incremental learning
        order = np.array(['2', '4', '0', '5', '3', '1'])
        for idx, lab in enumerate(self.targets.tolist()):
            self.targets[idx] = np.asarray(np.where(order == lab)[0],dtype = str)[0]

        # Loading the embeddings
        self.patch_embeddings = []
        for wsi_id in tqdm(self.images):
            npy_id = './local_data/embeddings/' + wsi_id + '.npy'
            npy_embeddings = np.load(npy_id)
            self.patch_embeddings.append(npy_embeddings)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        """Denotes the total number of samples/patients"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.patch_embeddings[idx]
        y = self.targets[idx]
        return x, y

############################################# MIL DATAGENERATOR ##################################################
class MILDataGenerator(object):

    def __init__(self, dataset, batch_size = 1, shuffle = False):

        """Data Generator object for MIL.
            Process a MIL dataset object to output batches of instances and its respective labels.
        Args:
          dataset: MIL datasetdataset object.
          batch_size: batch size (number of bags). Must be set to one.
          shuffle: whether to shuffle the bags (True) or not (False).

        Returns:
          MILDataGenerator object
        Last Updates: Pablo M. (01/2024)
        """
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_wsis = len(self.dataset.images)
        self.indexes = np.arange(self.n_wsis)
        self.targets = self.dataset.targets
        self.max_instances = 200
        self._idx = 0 # Seconday initializations

    def reset(self):
        self._idx = 0

    def __len__(self):
        N = self.n_wsis
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= self.n_wsis:
            self.reset()
            raise StopIteration()

        # Get samples of data frame to use in the batch
        idx_batch = self.indexes[self._idx]
        Y = int(self.dataset.targets[idx_batch])
        Y = np.expand_dims(np.array(Y), 0) # Bag-level label
        X = self.dataset.patch_embeddings[idx_batch]

        # # Patch limitation for reproducibility
        # if X.shape[0] > self.max_instances:
        #     idx_random = np.random.choice(X.shape[0],self.max_instances,replace=False)
        #     X = X[idx_random]

        # Update bag index iterator + return data
        self._idx += self.batch_size
        bag_img = torch.tensor(np.array(X).astype('float32')).cuda()
        bag_label = torch.tensor(Y, dtype=torch.int64).cuda()
        return bag_img, bag_label

class MILDataGenerator_und(object):

    def __init__(self, dataset, max_instances = 500, batch_size = 1, shuffle = False):

        """Data Generator object for MIL.
            Process a MIL dataset object to output batches of instances and its respective labels.
        Args:
          dataset: MIL datasetdataset object.
          batch_size: batch size (number of bags). It will be usually set to 1.
          shuffle: whether to shuffle the bags (True) or not (False).
          max_instances: maximum amount of instances allowed due to computational limitations.

        Returns:
          MILDataGenerator object
        Last Updates: Pablo Meseguer (//)
        """
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.data_frame))
        self.targets = self.dataset.targets
        self.max_instances = max_instances

        cnt = 0
        classes = np.unique(self.targets).tolist()
        n_class = np.zeros(len(classes), dtype=int)
        for class_id in classes:
            n_class[cnt] = sum(self.targets == class_id)
            cnt = cnt + 1
        max_class = classes[np.argmax(n_class)]
        min_class = classes[np.argmin(n_class)]
        idx_max_class = self.indexes[self.targets==max_class]
        self.n_minority = int(np.min(n_class))

        epochs = 50
        self.random_undersampling = np.zeros(shape = (epochs,2*self.n_minority), dtype = int)
        for it in range(0, epochs):
            self.random_undersampling[it, :self.n_minority] = np.random.choice(idx_max_class, size = self.n_minority, replace = False)
            self.random_undersampling[it, self.n_minority:] = self.indexes[self.targets==min_class]
            np.random.shuffle(self.random_undersampling[it])

        'Secondary Initializations'
        self._idx = 0
        self.epoch = 0
        # self.reset()

    def reset(self):
        self._idx = 0

    def __len__(self):
        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= 2*self.n_minority:
            self.epoch = self.epoch+1
            self.reset()
            raise StopIteration()

        # Get samples of data frame to use in the batch
        idx_und = self.indexes[self.random_undersampling[self.epoch,self._idx]]
        df_row = self.dataset.data_frame.iloc[idx_und]

        # Get bag-level label
        Y = np.expand_dims(np.array(int(df_row[self.dataset.classes])), 0)

        # Select instances from bag
        # ID = list(df_row[['slide_name']].values)[0]
        ID = list(df_row.values)[0]
        images_id = self.dataset.D[ID]

        # Memory limitation of patches in one slide
        if len(images_id) > self.max_instances:
            # images_id = random.sample(images_id, self.max_instances)
            images_id = images_id[0:self.max_instances]

        # Minimum number of patches in a slide (by precaution).
        # if len(images_id) < 30:
        #     images_id.extend(images_id)

        # Load images and include into the batch
        X = []
        for i in images_id:
            x = self.dataset.__getitem__(i)
            X.append(x)

        # Update bag index iterator
        self._idx += self.batch_size
        # return np.array(X).astype('float32'), np.array(Y).astype('float32')
        return torch.tensor(np.array(X).astype('float32')).cuda(), torch.tensor(np.array(Y).astype('long')).cuda()

############################################# IMAGE PROCESSING ##################################################
def image_normalization(self, x):
    x = cv2.resize(x, (self.input_shape[1], self.input_shape[2]))  # image resize
    x = x / 255.0  # intensity normalization
    if self.channel_first:  # channel first
        x = np.transpose(x, (2, 0, 1))
    x.astype('float32')  # numeric type
    return x

def norm_image(x, input_shape):
    # image resize
    x = cv2.resize(x, (input_shape[1], input_shape[1]))
    # intensity normalization
    x = x / 255.0
    # channel first
    x = np.transpose(x, (2, 0, 1))
    # numeric type
    x.astype('float32')
    return x

def image_transformation(im, border_value=1):
    im = np.transpose(im, (1, 2, 0))

    # Random index for type of transformation
    random_index = np.clip(np.round(random.uniform(0, 1) * 10 / 2), 1, 4)

    if random_index == 1 or random_index == 3:  # translation

        # Randomly obtain translation in pixels in certain bounds
        limit_translation = im.shape[0] / 4
        translation_X = np.round(random.uniform(-limit_translation, limit_translation))
        translation_Y = np.round(random.uniform(-limit_translation, limit_translation))
        # Get transformation function
        T = np.float32([[1, 0, translation_X], [0, 1, translation_Y]])
        # Apply transformation
        im_out = cv2.warpAffine(im, T, (im.shape[0], im.shape[1]),
                                borderValue=(border_value, border_value, border_value))

    elif random_index == 2:  # rotation

        # Get transformation function
        rotation_angle = np.round(random.uniform(0, 360))
        img_center = (im.shape[0] / 2, im.shape[0] / 2)
        T = cv2.getRotationMatrix2D(img_center, rotation_angle, 1)
        # Apply transformation
        im_out = cv2.warpAffine(im, T, (im.shape[0], im.shape[1]),
                                borderValue=(border_value, border_value, border_value))

    elif random_index == 4:  # mirroring

        rows, cols = im.shape[:2]
        src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        dst_points = np.float32([[cols - 1, 0], [0, 0], [cols - 1, rows - 1]])
        T = cv2.getAffineTransform(src_points, dst_points)

        im_out = cv2.warpAffine(im, T, (cols, rows), borderValue=(border_value, border_value, border_value))

    im_out = np.transpose(np.array(im_out), (2, 0, 1))
    return im_out


# Colour normalzation
# self.color_norm = color_norm
# if self.color_norm:
#     self.colour_normalizer = Colour_Normalization()
# class Colour_Normalization:
#     def __init__(self):
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x * 255)
#         ])
#
#         # Read reference image
#         target_image_path = './normalization/Reference_HUSC_5040_1.jpg'
#         target_image = tf.keras.preprocessing.image.load_img(target_image_path, target_size = (512, 512))
#         target_image = np.array(target_image)
#         target_image = Image.fromarray(target_image, 'RGB')
#
#         # Fit stain normalizer
#         torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend = 'torch')
#         torch_normalizer.fit(self.transform(target_image))
#         self.normalizer = torch_normalizer
#
#     def __call__(self, image, *args, **kwargs):
#         image = self.transform(image)
#         norm_image, _, _ = self.normalizer.normalize(I = image, stains = False)
#         norm_image = norm_image.numpy().astype(np.uint8)
#         return norm_image

############################################# FUTURE ##################################################
class MILDataset_v2(torch.utils.data.Dataset):

    def __init__(self, dir_images, data_frame, images_restriction, classes='GT', bag_id='HCUV',
                 input_shape=(3, 224, 224), data_augmentation = False, images_on_ram = True, channel_first = True):

        """Dataset object for MIL.
            Dataset object which aims to organize images and labels from a dataset in the form of bags.
            It reads the image file name from a csv with the patches detected as ROI
        Args:
          dir_images: (h, w, channels)
          data_frame: pandas dataframe with ground truth information.
                      Each bag is one raw, with 'bag_name' as identifier.
          classes: list of classes of interest in data_fame (i.e. ['G3', 'G4', 'G5'])
          input_shape: image input shape (channels first).
          data_augmentation: whether to perform data augmentation (True) or not (False).
          images_on_ram: whether to load images on ram (True) or not (False). Recommended for accelerated training.
        Returns:
          MILDataset object
        Past Updates: Rocío (19/04/2021)
        Last Updates: Pablo (30/09/2022)
        """

        self.dir_images = dir_images
        self.data_frame = data_frame
        self.classes = classes
        self.bad_id = bag_id
        self.data_augmentation = data_augmentation
        self.input_shape = input_shape
        self.images_on_ram = images_on_ram
        self.channel_first = channel_first
        self.device = torch.device("cuda:0")
        self.indexes = []
        self.indexes = np.array(self.indexes)
        self.image_restriction = images_restriction

        ##### READING THE PATCHES NAMES FROM THE ROI'S CSV #####
        self.images = pd.read_csv('./csv/ROI_Joint.csv', dtype = str)
        self.images = self.images.values.tolist()
        self.images = [item for sublist in self.images for item in sublist]

        # Organize bags in the form of dictionary: one key clusters indexes from all instances
        self.D = dict()
        for i, item in enumerate([ID.split('_')[0] + '_' + ID.split('_')[1] for ID in self.images]):
            if item not in self.D:
                self.D[item] = [i]
            else:
                self.D[item].append(i)
        self.indexes = np.arange(len(self.images))
        self.targets = self.data_frame[self.classes].values

        # Only patients in the set
        for ID in list(self.D.keys()):
            if not bool(np.isin(ID, list(self.data_frame.Patient.values))):
                del self.D[ID]

        # Image restriction
        self.selected_images = []
        for ID in self.D.keys():
            n_ims = min(len(self.D[ID]), self.image_restriction)
            self.D[ID] = random.choices(self.D[ID], k = min(len(self.D[ID]), n_ims))  # "Random" choise of the images
            self.selected_images = self.selected_images + self.D[ID]

        # Pre-allocate images, load and normalize images
        N = len(self.selected_images)
        if self.images_on_ram:
            self.X = np.zeros((N, input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)

            print('[INFO]: Loading images')
            for counter, i in enumerate(self.selected_images):
                print(str(counter + 1) + '/' + str(N))
                ID = self.images[i]
                if ID[0:4] == 'HCUV':
                    x = Image.open(os.path.join(self.dir_images, 'Images/', ID))  # Load image
                elif ID[0:4] == 'HUSC':
                    x = Image.open(os.path.join(self.dir_images, 'Images_Jose/', ID))  # Load image
                x = image_normalization(self, np.asarray(x))  # Normalization
                self.X[counter, :, :, :] = x

    def __len__(self):
        """Denotes the total number of samples/patients"""
        return len(self.indexes)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.images[np.int(self.indexes[index])]

        if self.images_on_ram:
            x = np.squeeze(self.X[self.indexes[index], :, :, :])
        else:
            x = Image.open(os.path.join(self.dir_images, ID))  # Load image
            x = np.asarray(x)
            x = image_normalization(self, x)  # Normalization

        if self.data_augmentation:  # data augmentation
            x = image_transformation(x)
        return x
class MILDataGenerator_v2(object):

    def __init__(self, dataset, batch_size=1, shuffle=False, max_instances=200):  # antes 250

        """Data Generator object for MIL.
            Process a MIL dataset object to output batches of instances and its respective labels.
        Args:
          dataset: MIL datasetdataset object.
          batch_size: batch size (number of bags). It will be usually set to 1.
          shuffle: whether to shuffle the bags (True) or not (False).
          max_instances: maximum amount of instances allowed due to computational limitations.

        Returns:
          MILDataGenerator object
        Last Updates: Pablo (30/009/22)
        """
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.data_frame))
        self.targets = self.dataset.targets
        self.max_instances = max_instances
        self._idx = 0

    def reset(self):
        self._idx = 0

    def __len__(self):
        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.dataset.data_frame):
            self.reset()
            raise StopIteration()

        # Get samples of data frame to use in the batch
        df_row = self.dataset.data_frame.iloc[self.indexes[self._idx]]
        Y = np.expand_dims(np.array(int(df_row[self.dataset.classes])), 0) # Bag label
        images_id = self.dataset.D[list(df_row.values)[0]] # Select instances from bag

        # Load images and include into the batch
        X = []
        for i in images_id:
            x = self.dataset.__getitem__(i)
            X.append(x)

        # Update bag index iterator
        self._idx += self.batch_size
        # return np.array(X).astype('float32'), np.array(Y).astype('float32')
        return torch.tensor(np.array(X).astype('float32')).cuda(), torch.tensor(np.array(Y).astype('long')).cuda()
