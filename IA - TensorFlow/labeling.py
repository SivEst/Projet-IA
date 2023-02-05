# On importe les librairies necessaires
import cv2 

import uuid

import os

import time

#On défini les images que l'on veut collecter
labels = ['zero', 'un', 'deux', 'trois', 'quatre', 'cinq']
#On defini le nombre d'image que l'on veut collecter
number_imgs = 20


#On crée un chemin dans le dossir IA où stocker les images que l'on va collecter

IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')
#Si le chemin vers IMAGES_PATH existe :
if not os.path.exists(IMAGES_PATH):
    #On cherche le système d'exploitation
    #Sur windows
    if os.name == 'nt':
         #On crée un dossier
         os.mkdir(IMAGES_PATH)
for label in labels:
    
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.mkdir(path)


for label in labels:
    #On se connecte à la 1 ere camera disponnible
    cap = cv2.VideoCapture(0)
    #On print les labels les uns apres les autres
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        #On print le nombre de photo qu'on prend
        print('Collecting image {}'.format(imgnum))
        #Capture une frame de la webcam
        ret, frame = cap.read()
        #Crée l'image et le placer dans le dossier corespondant
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        #Pour arreter le programme en appuiyant sur "q "
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#Ferme les cameras
cap.release()
cv2.destroyAllWindows()
