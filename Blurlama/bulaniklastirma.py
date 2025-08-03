# ÜÇ TÜR BULANIKLAŞTIRMA MEVCUT
# Ortalama Bulanıklaştırma
# Gauss Bulanıklaştırma
# Medyan Bulanıklaştırma

import cv2 #opencv Kütüphanesi
import matplotlib.pyplot as plt #matplotlib kütüphanesi pyplot modülü
import numpy as np numPy kütüphanesi

import warnings
warnings.filterwarnings("ignore")

#blurring(detayı azaltır, gürültüyü engeller)
img = cv2.imread("NYC.jpg") #Görüntüyü içeri aktarma
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #: OpenCV'nin BGR renk düzenini Matplotlib'in RGB düzenine çevirme
plt.figure() #Yeni bir matplotlib figürü oluşturma
plt.imshow(img) , plt.axis("off"), plt.title("orijinal"), plt.show()

"""
ortalama bulanıklaştırma yöntemi

"""
dst2 = cv2.blur(img, ksize = (3,3)) # 3x3 boyutunda bir bulanıklaştırma çekirdeği ile orijinal görüntüyü bulanıklaştırır.
plt.figure(), plt.imshow(dst2), plt.axis("off"), plt.title("Ortalama Blur")

"""
Gaussian blur

"""
gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7) #sigmaX parametresi, Gaussian çekirdeğinin standart sapmasını belirtir.
plt.figure, plt.imshow(gb), plt.axis("off"), plt.title("Gauss Blur")

"""
medyan blur

"""
mb = cv2.medianBlur(img, ksize = 3)
plt.figure(), plt.imshow(mb), plt.axis("off"), plt.title("Medyan Blur")

def gaussianNoise(image): 
    row, col, ch = image.shape
    mean = 0 # gaussian gürültüsü ortlaması 0 olarak belirlendi. Çünkü Gaussian gürültü genellikle ortlama etrafında dağılır.
    var = 0.05 # Varyans, gürültünün dağılımının genişliğini kontrol eder.daha yüksek varyans daha fazla gürültü demek. 
    sigma = var**0.5 # gaussian dağılımının şeklini belirler.
    
    gauss = np.random.normal(mean, sigma, (row, col, ch)) # row satır sayısı, col sütun sayısı, ch renk kanalları sayısı. 
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    
    return noisy

# içe aktar normalize et 
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
plt.figure()
plt.imshow(img), plt.axis("off"), plt.title("orijinal"), plt.show()

gaussianNoisyImage = gaussianNoise(img)
plt.figure(), plt.imshow(gaussianNoisyImage), plt.axis("off"), plt.title("Gauss Noisy"), plt.show()


#gauss blur
gb2 = cv2.GaussianBlur(gaussianNoisyImage, ksize = (3,3), sigmaX = 7)
plt.figure, plt.imshow(gb2), plt.axis("off"), plt.title("with Gauss Blur")



def saltPepperNoise(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    
    amount = 0.004
    
    noisy = np.copy(image)
    
    #salt beyaz
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords] = 1
    
    # pepper siyah
    num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords] = 0
    
    return noisy

spImage = saltPepperNoise(img)
plt.figure(), plt.imshow(spImage), plt.axis("off"),plt.title("SP image")

mb2 = cv2.medianBlur(spImage.astype(np.float32), ksize = 3)
plt.figure(), plt.imshow(mb2), plt.axis("off"), plt.title("with Medyan Blur")




















