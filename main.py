import cv2
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.models import  load_model
import numpy as np

model = load_model("D:\Face\GelasatauBotol\keras_model.h5")
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    roi_gray = cv2.resize(test_img, (224, 224))
    sample = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2RGB)
    img_pixels = image.img_to_array(sample)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    obj = ('gelas', 'botol')
    predicted_emotion = obj[max_index]
    cv2.putText(test_img, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Gelas atau Botol', resized_img)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows