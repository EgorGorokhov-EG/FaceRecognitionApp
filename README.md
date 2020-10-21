# FaceRecognitionApp

It's very simple pet-project app made with Tkinter for detection and recognition faces defined in database. Initially code using a model pretrained on some photos loaded by me. People on these photos are: David Bowie, Freddie Mercury, Jimi Hendrix and photos of random people just to define a class "Undefined"

# Demonstration
When program is opened this window appears: 
![alt text](DemoPhotos/0.PNG?raw=true)

To recognize person you need to select a photo first with "Select Photo" button and then click on "Recognize" button. For a demo I'm using a photo of me which model hasn't seen during training.

![alt text](DemoPhotos/1.PNG?raw=true)
![alt text](DemoPhotos/2.PNG?raw=true)

As you can see model defined me as "Undefined", and it's correct because it isn't trained to recognize me as a separate class.
Next I'm using a photo of Jimi Hendrix. This particular photo isn't in the training data, but Jimi is a defined person to recognize.

![alt text](DemoPhotos/3.PNG?raw=true)

In this time program worked correctly too recognizing Jimi Hendrix in this photo.

You can use your own photos to train model on. For this click on the "Load Photos" button and wait for some time while model embedding photos and training to classify them by their embeddings. When it's done you can select photo and recognize person in it as shown above.
