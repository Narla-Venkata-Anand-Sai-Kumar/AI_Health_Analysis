import tensorflow as tf
import cv2
import numpy as np


def classify(img):
    im = img
    lt = ["other","Bone","Brain","eye","kidney","chest","skin"] 
    im = cv2.resize(im,(52,52))
    model = tf.keras.models.load_model("all-in-one.h5",compile=False)
    result = model.predict(np.array([im]))
    a = np.argmax(result)
    c=""
    if a==0:
        return "Enter the medical Image"
    if a==1:
        c = bone_net(img)
    if a==2:
        c = brain_net(img)
    if a==3:
        c = Eye_net(img)
    if a==4:
        c = kidney_net(img)
    if a==5:
        c = chest_net(img)
    if a==6:
        c = skin_net(img)
    return c



def bone_net(img):
    img = cv2.resize(img,(224,224))
    model = tf.keras.models.load_model("Fracture_detection.h5",compile=False)
    result = model.predict(np.array([img]))
    op=""
    if result[0]<0.5:
        op="Fracture"   
    else:
        op="Normal"
    return op

def brain_net(img):
    lt = ['pituitary', 'notumor', 'meningioma', 'glioma']
    img = cv2.resize(img,(224,224))
    model = tf.keras.models.load_model("brain.h5",compile=False)
    result = model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]

def chest_net(img):
    lt = ['PNEUMONIA', 'NORMAL']
    img = cv2.resize(img,(224,224))
    model = tf.keras.models.load_model("chest_cls_model.h5",compile=False)
    result = model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]

def Eye_net(img):
    lt = ['glaucoma', 'normal', 'diabetic_retinopathy', 'cataract']
    img = cv2.resize(img,(224,224))
    model = tf.keras.models.load_model("Eye_diseases.h5",compile=False)
    result = model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]

def kidney_net(img):
    lt = ['Cyst', 'Tumor', 'Stone', 'Normal']
    img = cv2.resize(img,(224,224))
    model = tf.keras.models.load_model("kidney_stone.h5",compile=False)
    result = model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]

def skin_net(img):
    lt = ['pigmented benign keratosis', 'melanoma', 'vascular lesion', 'actinic keratosis', 'squamous cell carcinoma', 'basal cell carcinoma', 'seborrheic keratosis', 'dermatofibroma', 'nevus']
    img = cv2.resize(img,(224,224))
    model = tf.keras.models.load_model("skin_cancer.h5",compile=False)
    result = model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]