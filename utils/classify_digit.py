import cv2
import numpy as np
import tensorflow as tf

class Classify_Digit:
    def __init__(self, model_path, kernel_size=3, show_journey=False):
        self.model = tf.keras.models.load_model(model_path)
        self.kernel = np.ones((kernel_size,kernel_size), np.uint8)
        self.show_journey = show_journey
        self.classified_digits = None

    def classify(self, extracted_digits):
        # processed_digits = [(tf.keras.utils.normalize(cv2.erode(img.copy(), self.kernel, iterations=1), axis=1)).tolist() for img in extracted_digits]
        
        # # Doesn't  handle all black case
        # # predictions_distribution = self.model.predict(processed_digits)
        # # predicted_digits = [np.argmax(np.array(digit)) for digit in predictions_distribution]
        # # print(predicted_digits)

        # self.classified_digits = [0 if not np.any(np.array(digit)) else np.argmax(self.model.predict([digit])) for digit in processed_digits]
        # self.classified_digits = np.reshape(np.array(self.classified_digits), (9, 9))

        classified_digit = [[0 for j in range(9)] for i in range(9)]
        i, j = 0, 0
        
        if self.show_journey:
            rows, row = [], []
        
        for img in extracted_digits:
            test = (cv2.erode(img.copy(), self.kernel, iterations=1))

            if self.show_journey:
                row.append(cv2.copyMakeBorder(test.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, (255,255,255)))
            
            tf.keras.utils.normalize(test, axis=1)
            if not np.any(test):
                j = (j+1)%9
                if j == 0:
                    i += 1
                    if self.show_journey:
                        rows.append(np.concatenate(row, axis=1))
                        row = []
                continue
            pred = self.model.predict([test.tolist()])
            classified_digit[i][j] = np.argmax(pred[0])
            j = (j+1) % 9
            if j == 0:
                i += 1
                if self.show_journey:
                    rows.append(np.concatenate(row, axis=1))
                    row = []
        
        self.classified_digits = classified_digit

        if self.show_journey:
            cv2.imshow('extracted_digits', np.concatenate(rows))
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()