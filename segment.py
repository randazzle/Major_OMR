import cv2 
from matplotlib import pyplot as plt

image_file = "samples/fire.jpg"
img = cv2.imread(image_file)
base_img = img.copy()

#display_in_jupyternotebook
def display(im_path):
    dpi = 100
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

#display_in_jupyternotebook
def display2(im_path):
    dpi = 100
    im_data = plt.imread(im_path)
    height, width= im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("temp/gray.jpg", gray)
#display2("temp/gray.jpg")

blur = cv2.GaussianBlur(gray, (9,9), 0)
cv2.imwrite("temp/blur.jpg", blur)
#display2("temp/blur.jpg")

thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite("temp/thresh.jpg", thresh)
#display2("temp/thresh.jpg")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
cv2.imwrite("temp/kernel.jpg", kernel)
#display2("temp/kernel.jpg")

dilate = cv2.dilate(thresh, kernel, iterations=1)
cv2.imwrite("temp/dilate.jpg", dilate)
#display2("temp/dilate.jpg")

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cents[1]
cnts = sorted(cnts, key=lambda x:cv2.boundingRect(x)[0])

i = 1
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 350:
        roi = img[y:y+h, x:x+w]
        cv2.imwrite("temp/segment" + str(i)+ ".jpg", roi)
        cv2.rectangle(img, (x, y), (x+w, y+h), (36, 266, 12), 2)
        #display("temp/segment"+str(i)+".jpg")
        i = i+1 
cv2.imwrite("temp/segmented.jpg", img)
#display("temp/segmented.jpg")
totalimages = i

segmentedimg = []
for x in range(1,i):
    segmentedimg.append("temp/segment"+str(x)+".jpg")
    
print(segmentedimg)