import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def blurring(image):
    k_size = (7, 7)
    blurred_image = cv2.GaussianBlur(image, k_size, 0)
    return blurred_image

def segmenting(image):
    char_list = []
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if h < 30 or w < 10 :
            continue
        curr_char = image[y:y+h, x:x+w]
        char_list.append(curr_char)
    return char_list

def ocr_on_segmented_characters(segmented_characters):
    custom_config = r'--oem 3 --psm 10' # adjust configuration as per your requirement
    recognized_text = ""
    for char_image in segmented_characters:
        char_text = pytesseract.image_to_string(char_image, config=custom_config)
        recognized_text += char_text.strip()
    return recognized_text
    
def number_plate():
    width = 800
    height = 400
    img = cv2.imread(r"C:\Users\KIIT01\Desktop\SpeedTrackingUpdated\license-plate.jpg")
    img = cv2.resize(img, (width, height))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    plates = cascade.detectMultiScale(gray, 1.2, 5)
    print('Number of detected license plates:', len(plates))

    for (x,y,w,h) in plates:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        gray_plates = gray[y:y+h, x:x+w]
        color_plates = img[y:y+h, x:x+w]
        characters = segmenting(thresholding(color_plates))
        text = ocr_on_segmented_characters(characters)

        print(text)
        cv2.imwrite('Numberplate.jpg', gray_plates)
        cv2.imshow('Number Plate', gray_plates)
        cv2.imshow('Number Plate Image', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    number_plate()



