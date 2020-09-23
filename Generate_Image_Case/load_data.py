from google.colab import drive
drive.mount('/content/drive')

from PIL import Image
import sys, os, urllib.request, tarfile, cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing import image

def convert(x, size=96):
    result = []
    for i in range(len(x)):
        img = cv2.resize(x[i],(size, size))
        result.append(img)
        
    return np.array(result)

class AD:
    def __init__(self, download_dir, path):
        self.path = "data/"

        if not os.path.exists(download_dir):
            os.mkdir(download_dir)

        # download file
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (source_path,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        source_path = path
        dest_path = os.path.join(download_dir, "data.tar.xz")
        urllib.request.urlretrieve(source_path, filename=dest_path, reporthook=_progress)
        # untar
        with tarfile.open(dest_path, "r:xz") as tar:
            tar.extractall(self.path)

    def load_images(self, path, num):
        result = []
        for i in range(num):
            if i < 10:
                img = Image.open(self.path + path + "00" + str(i) + ".png")
            elif i < 100:
                img = Image.open(self.path + path + "0" + str(i) + ".png")
            else:
                img = Image.open(self.path + path + str(i) + ".png")
            img = image.img_to_array(img)
            img = cv2.resize(img,(224,224))
            result.append(img)
        return np.array(result)

print("\nHazelnut data download...")
Hazelnut = AD("./ad", "ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/hazelnut.tar.xz")
hazelnut_train = Hazelnut.load_images("hazelnut/train/good/", 391)
hazelnut_test_normal = Hazelnut.load_images("hazelnut/test/good/", 40)
hazelnut_test_anomaly = Hazelnut.load_images("hazelnut/test/crack/", 18)
hazelnut_test_anomaly = np.vstack((hazelnut_test_anomaly, Hazelnut.load_images("hazelnut/test/cut/", 17)))
hazelnut_test_anomaly = np.vstack((hazelnut_test_anomaly, Hazelnut.load_images("hazelnut/test/print/", 17)))
hazelnut_test_anomaly = np.vstack((hazelnut_test_anomaly, Hazelnut.load_images("hazelnut/test/hole/", 18)))

hazelnut_train /= 255
hazelnut_test_normal /= 255
hazelnut_test_anomaly /= 255

x_train = convert(hazelnut_train)
x_test_normal = convert(hazelnut_test_normal)
x_test_anomaly = convert(hazelnut_test_anomaly)
print(x_test_normal.shape)
plt.imshow(x_test_normal[1]),plt.show()

np.save('drive/My Drive/product_X', x_test_normal[:20])

