import cv2
import easyocr
import numpy as np
import random
from PIL import Image
import tempfile

list = ['D:/Project/TuanHoang/data/output/license_plate_car_0.7277054786682129.jpg',
        'D:/Project/TuanHoang/data/output/license_plate_car_0.7787944674491882.jpg',
        'D:/Project/TuanHoang/data/output/license_plate_car_0.8018612861633301.jpg']

# Initialize the reader
reader = easyocr.Reader(['en'])

def set_image_dpi(input_path):
    dpi = (1200, 1200)
    weight = 300
    height = 100
    output_path = input_path + "_dpi_changed.jpg"
    with Image.open(input_path) as img:
        resize_img = img.resize((weight, height))
        resize_img.save(output_path, dpi=dpi)
    print(f"Ảnh đã được lưu với DPI {dpi} tại {output_path}")
    return output_path

def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)


# step 1: Resize Dpi the image
path_image = random.choice(list)
path_image = set_image_dpi(path_image)
image = cv2.imread(path_image)

# step 2: Normalization
norm_img = np.zeros((image.shape[0], image.shape[1]))
image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
# step 3: remove noise
# image = remove_noise(image)
# step 4: sharpening
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(image)
sharpening_kernel = np.array([[ 0, -0.25,  0],
                              [-0.25,  2, -0.25],
                              [ 0, -0.25,  0]])
image = cv2.filter2D(image, -1, sharpening_kernel)

result = reader.readtext(image)[0]
print(result)

top_left = result[0][0]
top_right = result[0][1]
bottom_right = result[0][2]
bottom_left = result[0][3]

print("Top left:", top_left)
print("Top right:", top_right)
print("Bottom right:", bottom_right)
print("Bottom left:", bottom_left)
x1, y1 = int(top_left[0]), int(top_left[1])
x2, y2 = int(bottom_right[0]), int(bottom_right[1])
cropped_image = image[y1:y2, x1:x2]
result_crop = reader.readtext(cropped_image)[0]
print(result_crop)  

cv2.imshow("Cropped Text Region", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()