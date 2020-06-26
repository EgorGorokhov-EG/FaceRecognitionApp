import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import imutils
from imutils.face_utils import FaceAligner
import dlib
from catboost import CatBoostClassifier, Pool
import numpy as np
import re
from PIL import ImageTk, Image


class FaceRecognizerApp(tk.Tk):
    def __init__(self):
        current_dir = os.getcwd()
        #  All model needed for a face recognition
        self.detectorConfig = current_dir + r"\pretrained models\deploy.prototxt"
        self.detectorFile = current_dir + r"\pretrained models\res10_300x300_ssd_iter_140000.caffemodel"
        self.predictorFile = current_dir + r"\pretrained models\shape_predictor_68_face_landmarks.dat"
        self.embedderFile = current_dir + r"\pretrained models\nn4.v2.t7"

        super(FaceRecognizerApp, self).__init__()

        #  Parameters of the main window
        self.title("Recognizer")
        self.geometry("800x600")
        self.configure(background='black')

        # Elements in the window
        # Label Frames
        self.label_photo = tk.LabelFrame(self, text="Current Photo", padx=10, pady=10, background='black', foreground="white")
        self.label_photo.grid(column=0, row=1, sticky='W')
        self.label_loading = tk.LabelFrame(self, text="Load Photos", padx=10, pady=10, background='black', foreground="white")
        self.label_loading.grid(column=1, row=0, sticky='N')
        self.label_display = tk.LabelFrame(self, text="Display", background='black', foreground="white")
        self.label_display.grid(column=0, row=0)

        #  Labels
        self.loading_stat = tk.Label(self.label_loading, text="Default photos loaded", background='black', foreground="white")
        self.loading_stat.grid(column=0, row=1)
        self.prediction = tk.Label(self.label_photo, text="", anchor='center', background='black', foreground="white")
        self.prediction.grid(column=0, row=1)

        self.display = tk.Canvas(self.label_display, background='black')
        self.display.grid(column=0, row=0)

        #  Declaring all images that will be used
        self.current_photo = None
        self.img = None
        self.button_recognize_img = ImageTk.PhotoImage(Image.open(r"D:\Python Projects\Face Recognition\GUI\Button.png"))
        self.button_select_img = ImageTk.PhotoImage(Image.open(r"D:\Python Projects\Face Recognition\GUI\Button1.png"))

        self.names = []  # List of all names in dataset

        self.button()
        self.button_lp()
        self.button_recognize()

    def button(self):
        btn = tk.Button(
            self.label_photo,
            command=self.choose_photo,
            text="Select Photo",
            background='black',
            foreground='white'
        )
        btn.grid(column=0, row=0)

    def button_lp(self):
        btn = tk.Button(
            self.label_loading,
            text="Load photos",
            command=self.load_photos,
            background='black',
            foreground='white'
        )
        btn.grid(column=0, row=0)

    def button_recognize(self):
        btn = tk.Button(
            self.label_photo,
            command=self.recognize,
            text="Recognize",
            background='black',
            foreground='white'
        )
        btn.grid(column=1, row=0)

    def load_photos(self):
        self.loading_stat.configure(text="Loading custom photos.\nPlease wait...")
        dir_path = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Folder")
        self.convert_to_vecs(dir_path)

    def choose_photo(self):

        self.current_photo = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select File",
            filetypes=(("Photo", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )

        file = Image.open(self.current_photo)
        width, height = file.size
        if width > 1000 or height > 1000:
            res_width = int(width/8)
            res_height = int(height/8)
        else:
            res_width = width
            res_height = height
        file = file.resize((res_width, res_height))
        self.img = ImageTk.PhotoImage(file)
        self.display.configure(width=res_width, height=res_height)
        self.display.create_image(0, 0, anchor='nw', image=self.img)

    def recognize(self):
        if self.current_photo is not None:
            pred, probas = self.fit_predict(self.current_photo)
            prediction = pred[0][0].split("_")
            name = ''
            for i in range(len(prediction)):
                name += prediction[i].capitalize()
                name += ' '

            self.prediction.configure(text=name)
        else:
            messagebox.showerror("No Photo Selected", "No photo selected. Please, select a file first.")

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
            print(img_name)
            label = img_name.split('.')[0]
            label = re.sub('[^a-z_]', '', label)

            img = cv2.imread(images_path + '/' + img_name)
            img = imutils.resize(img, width=600)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rect = detector_fa(gray, 3)

            #  make sure that face is detected or search again longer
            if len(rect) != 0:
                rect = rect[0]
            else:
                rect = detector_fa(gray, 5)[0]

            face_aligned = aligner.align(img, gray, rect)
            face_blob = cv2.dnn.blobFromImage(
                face_aligned,
                scalefactor=1. / 255,
                size=(96, 96),
                mean=(0, 0, 0),
                swapRB=True
            )

            embedder.setInput(face_blob)
            vec = embedder.forward()
            vecs[:, i] = vec
            print("{} images processed".format(i + 1))
            i += 1
            labels.append(label)

        model = CatBoostClassifier(verbose=False)
        data_Pool = Pool(vecs.T, labels)
        model.fit(data_Pool)
        model.save_model('trained_model')

        labels = list(set(labels))
        for label in labels:
            name = label.split('_')
            for i in range(len(name)):
                self.names.append(name[i].capitalize() + " ")

        print("Converting complete")
        self.loading_stat.configure(text="Custom photos loaded")

    def fit_predict(self, img_path):
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

        img = cv2.imread(img_path)
        img = imutils.resize(img, width=600)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = detector_fa(gray, 3)

        #  make sure that face is detected or search again longer
        if len(rect) != 0:
            rect = rect[0]
        else:
            rect = detector_fa(gray, 5)[0]

        face_aligned = aligner.align(img, gray, rect)

        '''
        (x, y, w, h) = helpers.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('face', img)
        cv2.waitKey(0)
        '''

        face_blob = cv2.dnn.blobFromImage(
            face_aligned,
            scalefactor=1. / 255,
            size=(96, 96),
            mean=(0, 0, 0),
            swapRB=True
        )

        embedder.setInput(face_blob)
        vec = embedder.forward()

        model = CatBoostClassifier()
        model.load_model('trained_model')
        pred = model.predict(vec)
        probas = model.predict_proba(vec)

        return pred, probas


if __name__=='__main__':
    root = FaceRecognizerApp()
    root.mainloop()
