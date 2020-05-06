import cv2
import imutils
from imutils.face_utils import helpers, FaceAligner
import dlib
from catboost import CatBoostClassifier, Pool
import numpy as np
import os
import re


class Facer:

    def __init__(self):
        self.detectorConfig = r"D:\Python Projects\Face Recognition\pretrained models\deploy.prototxt"
        self.detectorFile = r"D:\Python Projects\Face Recognition\pretrained models\res10_300x300_ssd_iter_140000.caffemodel"
        self.predictorFile = r"D:\Python Projects\Face Recognition\pretrained models\shape_predictor_68_face_landmarks.dat"
        self.embedderFile = r"D:\Python Projects\Face Recognition\pretrained models\nn4.v2.t7"

    def convert_to_vecs(self, images_path):
        """
        Finds faces in the images and transform them into 128-dimensional vectors
        Arguments:
        images_path - path to the folder with photos to transform. Each photo must be named like this: NUM_NAME.jpg
        Returns:
        vecs -- array with embeddings of given faces. Shape(128, num_of_examples)
        labels -- array with names corresponding to embeddings. Shape(1, num_of_examples)
        """

        num_examples = len(os.listdir(images_path))
        vecs = np.zeros((128, num_examples))  # array to store all face embeddings
        labels = list()  # labels for these embeddings

        detector_fa = dlib.get_frontal_face_detector()  # face detector for alignment
        embedder = cv2.dnn.readNetFromTorch(self.embedderFile)  # model to create embeddings from faces
        predictor = dlib.shape_predictor(self.predictorFile)  # model for creating face landmarks(not used ow)
        aligner = FaceAligner(predictor)  # custom face aligner made by Adrian Rosebrock

        i = 0
        for img_name in os.listdir(images_path):
            label = img_name.split('.')[0]
            label = re.sub('[^a-z_]', '', label)

            img = cv2.imread(images_path + '/' + img_name)
            img = imutils.resize(img, width=600)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rect = detector_fa(gray, 5)[0]

            face_aligned = aligner.align(img, gray, rect)
            '''
            cv2.imshow('photo', face_aligned)
            cv2.waitKey(0)
            '''
            face_blob = cv2.dnn.blobFromImage(face_aligned,
                                              scalefactor=1. / 255,
                                              size=(96, 96),
                                              mean=(0, 0, 0),
                                              swapRB=True)
            embedder.setInput(face_blob)
            vec = embedder.forward()
            vecs[:, i] = vec
            i += 1
            labels.append(label)

        model = CatBoostClassifier(verbose=False)
        data_Pool = Pool(vecs.T, labels)
        model.fit(data_Pool)
        model.save_model('trained_model')

        return vecs, labels

    def fit_predict(self, img):
        """
        Create embedding for given image
        Arguments:
        img -- image to get embedding from(array from cv2.imread())
        Return:
        pred -- predicted name for given image
        probas -- probabilities for every class
        """

        detector_fa = dlib.get_frontal_face_detector()
        embedder = cv2.dnn.readNetFromTorch(self.embedderFile)
        predictor = dlib.shape_predictor(self.predictorFile)
        aligner = FaceAligner(predictor)

        img = imutils.resize(img, width=600)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = detector_fa(gray, 4)[0]

        face_aligned = aligner.align(img, gray, rect)

        '''
        (x, y, w, h) = helpers.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('face', img)
        cv2.waitKey(0)
        '''

        face_blob = cv2.dnn.blobFromImage(face_aligned,
                                          scalefactor=1. / 255,
                                          size=(96, 96),
                                          mean=(0, 0, 0),
                                          swapRB=True)

        embedder.setInput(face_blob)
        vec = embedder.forward()

        model = CatBoostClassifier()
        model.load_model('trained_model')
        pred = model.predict(vec)
        probas = model.predict_proba(vec)

        return pred, probas
