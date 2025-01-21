import cv2 
import face_recognition
import numpy as np
import sys


def loading_image_example():
    img = cv2.imread('data/Lionel Messi.jpg')    

    if img is None:
        print('Could not open or find the image')
        sys.exit()
    else:
        print('Image is loaded')


    # convert to rgb
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_encodings = face_recognition.face_encodings(rgb_img)

    # Show the image 
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_example():
    img1 = cv2.imread('data/Lionel Messi.jpg')    
    img2 = cv2.imread('data/Lionel Messi 2.png')    
    img3 = cv2.imread('data/Elon Musk.jpg')

    if img1 is None or img2 is None or img3 is None:
        print('Could not open or find the image')
        sys.exit()
    else:
        print('Image is loaded')

    # convert to rgb
    rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    rgb_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

    img1_encodings = face_recognition.face_encodings(rgb_img1)[0]
    img2_encodings = face_recognition.face_encodings(rgb_img2)[0]
    img3_encodings = face_recognition.face_encodings(rgb_img3)[0]

    # Compare the faces
    results = face_recognition.compare_faces([img1_encodings], img2_encodings)
    print(f"Image 1 and Image 2: {results} should be True")
    results2 = face_recognition.compare_faces([img1_encodings], img3_encodings)
    print(f"Image 1 and Image 3: {results2} should be False")

    # Show the image 
    cv2.imshow('Image 1', img1)
    cv2.imshow('Image 2', img2)
    cv2.imshow('Image 3', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    # loading_image_example()
    compare_example()
