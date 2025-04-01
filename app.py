import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False
PREDICT = True
MODEL = load_model('best_model.h5')
LABELS = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}
WINDOWSIZEX, WINDOWSIZEY = 640, 480
BOUNDRYINC = 5
img_counter = 0

pygame.init()
FONT = pygame.font.SysFont('freesansbold.ttf', 18)
DISPLAY = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption('Handwriting Recognition')

iswriting = False
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION:
            if iswriting:
                x, y = event.pos
                pygame.draw.circle(DISPLAY, WHITE, (x, y), 4, 0)
                number_xcord.append(x)
                number_ycord.append(y)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)
                min_x = max(0, number_xcord[0] - BOUNDRYINC)
                max_x = min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
                min_y = max(0, number_ycord[0] - BOUNDRYINC)
                max_y = min(WINDOWSIZEY, number_ycord[-1] + BOUNDRYINC)

                number_xcord = []
                number_ycord = []

                if max_x > min_x and max_y > min_y:
                    img_arr = np.array(pygame.PixelArray(DISPLAY))[min_x:max_x, min_y:max_y].T.astype(np.float32)

                    if IMAGESAVE:
                        cv2.imwrite(f'image_{img_counter}.png', img_arr)
                        img_counter += 1

                    if PREDICT:
                        if img_arr.size > 0:
                            image = cv2.resize(img_arr, (28, 28))
                            image = np.pad(image, ((10, 10), (10, 10)), mode='constant', constant_values=0)
                            image = cv2.resize(image, (28, 28))
                            image = image / 255.0
                            image_reshaped = image.reshape(1, 28, 28, 1)
                            predict_result = MODEL.predict(image_reshaped)
                            label_index = np.argmax(predict_result)
                            label = str(LABELS[label_index])

                            textsurface = FONT.render(label, True, RED)
                            textrect = textsurface.get_rect()
                            textrect.left, textrect.bottom = min_x, min_y - 5
                            DISPLAY.blit(textsurface, textrect)
                        else:
                            print("Warning: Captured image array is empty.")
                else:
                     print("Warning: Invalid bounds for image extraction.")

        if event.type == KEYDOWN:
            if event.key == K_n:
                DISPLAY.fill(BLACK)

    pygame.display.update()

            














