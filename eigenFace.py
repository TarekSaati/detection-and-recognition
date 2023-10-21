import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        img = img[y:y+h, x:x+w]  # Crop the face
    img = np.resize(img, (100, 100))
    img = img / 255.0  # Normalize
    return img

# # Step 1: Prepare training data
# # Assuming you have some face images stored in a directory "faces"
# # Each subdirectory in "faces" represents a person
# faces_dir = r'G:\datasets\faces\FaceData\FaceDataset'
# faces_dirs = os.listdir(faces_dir)
# faces, labels = [], []

# for label in range(800):
#     subject_dir_path = faces_dir + "/" + str(label)
#     subject_images_names = os.listdir(subject_dir_path)
#     labelfaces = []
#     for image_name in subject_images_names:
#         image_path = subject_dir_path + "/" + image_name
#         image = preprocess(image_path)
#         labelfaces.append(image)
#     if len(labelfaces) > 40:
#         labels.extend([label] * len(labelfaces))
#         faces.extend(labelfaces)

# np.save('TrainingFacesArr', np.array(faces))
# np.save('TrainingLabelsArray', np.array(labels))

faces = np.load('./TrainingFacesArr.npy')
labels = np.load('./TrainingLabelsArray.npy')
# Step 2: Train the EigenFace Recognizer
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.train(faces, labels)

# Step 3: Use the model to recognize faces
# Assuming test_img is your test image
test_img = preprocess('path/to/image.jpg')
label, confidence = face_recognizer.predict(test_img)
print(label, confidence)