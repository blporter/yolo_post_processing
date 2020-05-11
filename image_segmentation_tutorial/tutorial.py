from skimage.color import rgb2gray
from skimage import transform
import matplotlib.pyplot as plt


def plot_image(image_name):
    image = plt.imread(image_name)
    plt.imshow(image)
    plt.show()

    gray = rgb2gray(image)
    gray = transform.rescale(gray, 1 / 2)

    # Take the mean of the pixel values and use that for calculating thresholds.
    gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
    mean = gray_r.mean()
    for i in range(gray_r.shape[0]):
        if gray_r[i] > mean * 1.5:
            gray_r[i] = 4
        elif gray_r[i] > mean:
            gray_r[i] = 3
        elif gray_r[i] > mean / 1.5:
            gray_r[i] = 2
        elif gray_r[i] > mean / 3:
            gray_r[i] = 1
        else:
            gray_r[i] = 0

    gray = gray_r.reshape(gray.shape[0], gray.shape[1])
    plt.imshow(gray, cmap='gray')
    plt.show()


plot_image('normal_chest.jpg')
plot_image('pneumonia_chest.jpg')
