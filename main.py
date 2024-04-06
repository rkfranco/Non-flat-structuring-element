# Aluno: Rodrigo Kapulka Franco
import cv2  # pip install opencv-python
import numpy as np
from matplotlib import pyplot as plt

image_path = 'imgs/mama.png'
thresh = 25

def show_imgs(items):
    for item in items:
        plt.imshow(item[0], 'gray')
        plt.title(item[1])
        plt.show()


if __name__ == '__main__':
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    top_hat_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, np.ones((5, 5), np.uint8))
    processed_img = cv2.threshold(top_hat_img, thresh, 255, cv2.THRESH_BINARY)[1]

    images = [(img, 'Imagem original'),
              (top_hat_img, 'Top hat'),
              (processed_img, 'Imagem processada - Non-flat structuring element')]

    show_imgs(images)
    cv2.imwrite("./imgs/top_hat.png", top_hat_img)
    cv2.imwrite("./imgs/processed_img.png", processed_img)
