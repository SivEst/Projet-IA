from cvzone.HandTrackingModule import HandDetector
from tkinter import *
import time, cv2, os, shutil, sys

folders = ["train", "test", "data"] # Les deux dossiers
keys_keyboard = ["&","é", '"', "'", "(", "-", "è", "_", "ç", "0"] # Correspond aux touches 1 2 3 4 5 6 7 8 9 0 du clavier
real_keys_keyboard = ["1", "2", "3" , "4", "5" , "6" , "7" , "8" , "9" , "0"]

for folder in folders: # Pour chaque élément dans le liste folders
    try: #Si le dossier n'est pas présent on créer le dossier
        os.listdir(f"./{folder}")
    except:
        print(f"Dossier {folder} introuvable, création en cours du dossier {folder} ...")
        os.mkdir(f"./{folder}")

nb_classes = 0
classe_names = []

while nb_classes <= 0 or nb_classes > 10: # Tant que l'utilisateur n'as pas rentrée au moin une classe
    try: # Si l'utilisateur n'as pas rentrée un chiffre ou un nombre
        nb_classes = int(input("Veuillez saisir le nombre de classes que vous voulez créer (10 maximum) : " ))
    except:
        print("ERREUR : Veuillez saisir un nombre !!")

for i in range(nb_classes): 
    classe_names.append(input(f"Veuillez sasir le nombre de la classe n°{i+1}: ")) # Entre le nom de chaque classes un par un

print("Voici les touches du clavier corréspondant aux classes")
for i in range(len(classe_names)):
    print(f"{real_keys_keyboard[i]} : {classe_names[i]}")

try:
    for classe in classe_names:
        os.mkdir(f"./data/{classe}")
except:
    pass

# Capture du flux video de la weebcam avec une résultion de 720p
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Detection de la main
detector = HandDetector(maxHands=1, mode=False)

# Espace entre la main et les bords de l'image (en pixel)
offset = 20


def sort():
    labels = os.listdir("./data")
    for label in labels:
        data = os.listdir("./data/" + label)
        split = int(len(data) * 0.8)
        training_data = data[:split]
        validation_data = data[split:]
        for element in training_data:
            shutil.copy("./data/" + label + "/" + element, "./train/" + element)
        for element in validation_data:
            shutil.copy("./data/" + label + "/" + element, "./test/" + element)

def imgToXml():
    train = os.listdir("./train")
    test = os.listdir("./test")
    for element in train:
        for classe in classe_names:
            if classe in element:
                folder = classe
        img_open = cv2.imread("./train/" + element)
        dimensions = img_open.shape
        a, b = dimensions[0], dimensions[1]
        text = f"""<annotation><folder>five</folder><filename>{element}</filename><path></ path><source><database>Unknown</database></source><size><width>{a}</width><height>{b}</height><depth>3</depth></size><segmented>0</segmented><object><name>{folder}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>{a}</xmax><ymax>{b}</ymax></bndbox></object></annotation>"""
        fichier = open("./train/" + element.split(".")[0] + "xml", "w")
        fichier.write(text)
        fichier.close()
    for element in test:
        for classe in classe_names:
            if classe in element:
                folder = classe
        img_open = cv2.imread("./test/" + element)
        dimensions = img_open.shape
        a, b = dimensions[0], dimensions[1]
        text = f"""<annotation><folder>five</folder><filename>{element}</filename><path></ path><source><database>Unknown</database></source><size><width>{a}</width><height>{b}</height><depth>3</depth></size><segmented>0</segmented><object><name>{folder}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>{a}</xmax><ymax>{b}</ymax></bndbox></object></annotation>"""
        fichier = open("./test/" + element.split(".")[0] + ".xml", "w")
        fichier.write(text)
        fichier.close()



# Boucle infinie pour capturer chaque image du flux de la weebcam
while True:
    sucess, img = cap.read() # La variable sucess est un booléan qui va dire si oui ou non il a reussi a récuperer une image du flux de la weebcam
    imgFinal = img.copy() # On copy l'image pour la garder vierge
    hands, img = detector.findHands(img, draw=True) # On détècte les mains présente sur l'image et on l'assigner à la variable hands, la variable img est l'image de base avec la main tracer

    if hands: # Si une main est détécter 
        hand = hands[0] # On prend le première main dans toutes les mains qu'il aura détécter (même si on a décider aurapavant de scanner qu'une seule main à la fois)
        x, y, w, h = hand["bbox"] # On récupère les cordonnées de la main par rapport à l'image
        imgCrop = imgFinal[y-offset:y + h+offset, x-offset:x + w+offset] # On redimensionne l'image pour avoir juste la main mais à partir de l'image vierge pour ne pas avoir les traits tracer
        try:
            cv2.imshow("ImageFinal", img) # On affiche l'image
        except:
            pass

    key_pressed = cv2.waitKey(1) # On créer une variable qui récupère les touches appuyer sur le clavier 
    if key_pressed == ord("s"): # Si la touche "s" est préssée on enregistre l'image de la main redimensionner
        img_name = "Imgae_" + str(round(time.time()*1000)) + "(" + folder + ").jpg" 
        cv2.imwrite("./data/" + folder + "/" + img_name, imgCrop)

        
        img_open = cv2.imread("./data/" + folder + "/" + img_name)
        dimensions = img_open.shape
        a, b = dimensions[0], dimensions[1]
        text = f"""<annotation><folder>five</folder><filename>{img_name}</filename><path></ path><source><database>Unknown</database></source><size><width>{a}</width><height>{b}</height><depth>3</depth></size><segmented>0</segmented><object><name>{folder}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>{a}</xmax><ymax>{b}</ymax></bndbox></object></annotation>"""

    for i in range(len(keys_keyboard)):
        if ord(keys_keyboard[i]) == key_pressed:
            folder = classe_names[i]
            print(f"Les images vont être enregistrer pour la classe {classe_names[i]}")
    if key_pressed == ord("e"):
        sort()
        imgToXml()
        sys.exit()