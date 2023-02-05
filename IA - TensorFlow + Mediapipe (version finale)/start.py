from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math, time, cv2,webbrowser, random, pygame,os, numpy as np, tkinter as tk

# La librairie cvzone utilise mediapipe

######################################## INFO ###################################
# Vous pouvez ajouter des musiques au format mp3 dans le dossier songs
# Credit: Clément MAZEAU, Olivier CLAVIER, Julia GROSSI
######################################## INFO ###################################


##### TKINTER ######
fenetre = tk.Tk()
fenetre.title("IA créer par Clément MAZEAU, Olivier CLAVIER et Julia GROSSI")

# Empêche l'utilisateur de redimensionner la fenêtre tkinter
fenetre.resizable(width=False, height=False)
 
# Ouverture de l'image pour les instructions
image = Image.open('./images/ok(resize).png')

# Conversion de l'image au format Tkinter
photo = ImageTk.PhotoImage(image)

# Label pour le text instructions
text_instruction = tk.Label(fenetre, text="Instruction pour le bon fonctionnement de l'IA :", font=("Arial", 15))
text_instruction.grid(column=0, row=0, padx=(20,0), pady=(20,0))
 
# Label pour afficher l'image d'instruction
img_instruction = tk.Label(fenetre, image=photo, borderwidth=1, relief="solid")
img_instruction.grid(column=0,row=1)

# Label pour afficher le flux video de la weebcam
weebcam_label = tk.Label(fenetre, borderwidth=1, relief="solid")
weebcam_label.grid(column=1,row=1, padx=(0,20), pady=(0,10), columnspan=2)

# Label pour afficher la prediction de l'IA
text_prediction = tk.Label(fenetre, font=("Arial", 15))
text_prediction.grid(column=1, row= 0)

# Label pour afficher le texte d'ouverture
text_ouverture = tk.Label(fenetre, font=("Arial", 15))
text_ouverture.grid(column=2, row= 0)

# Label pour afficher les informations
text = """Informations :
- Si vous voulez fermer la fenêtre appuyer sur la touche « echap »
- Cette IA est créée avec notre propre base de données et entrainer avec notre propre neurone artificiel 
- Vous pouvez ouvrir une page YouTube sur un navigateur en faisant le signe 5 de la main et en le maintenant pendant 5 secondes 
- Vous pouvez également lancer une musique aléatoire en faisant le signe 0 de la main et en le maintenant pendant 5 secondes
"""
text_info = tk.Label(fenetre, text=text, borderwidth=1, relief="solid" , font=("Arial", 10), justify="left")
text_info.grid(column=0, row=2, columnspan=3, pady=(0,20))


##### IA #######
# Detecteur pour les mains
detector = HandDetector(maxHands=1, mode=False)
# Charge le modèle
classifier = Classifier("./models/keras_model.h5", "./models/labels.txt")


# Capture du flux video de la weebcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Format 1280 x 720 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# (désolé)
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# VARIABLES
old_hand_posture = 6
current_time = 0
music_is_playing = False

# Liste de toutes les musiques disponibles en local
songs = os.listdir("./songs")
song_current = ""

pygame.mixer.init()


def waiting(hand_posture):
    global old_hand_posture, current_time, music_is_playing, song_current
    if old_hand_posture != hand_posture:
        current_time = time.time_ns() // 1_000_000
    else:
        # Si la posture de la main correspond à un cinq
        if hand_posture == 5:
            time_relase = (5 -((time.time_ns() // 1_000_000) - current_time)/1000)
            if time_relase < 0:
                current_time = time.time_ns() // 1_000_000
                webbrowser.open(url)
            else:
                text = "Ouverture du lien YouTube dans " + str(format(time_relase, '.1f') + " s")
                text_ouverture.config(text=text)

        # Si la posture de la main correspond à un zero    
        elif hand_posture == 0:
            time_relase = (5 -((time.time_ns() // 1_000_000) - current_time)/1000)
            if time_relase < 0:
                current_time = time.time_ns() // 1_000_000
                if music_is_playing == False:
                    song = random.choice(songs)
                    pygame.mixer.music.load("./songs/"+song)
                    pygame.mixer.music.play()
                    song_current = song
                    music_is_playing = True
                else:
                    pygame.mixer.music.stop()
                    music_is_playing = False
            else:
                if music_is_playing == False:
                    text = "Lecture de la musique dans " + str(format(time_relase, '.1f') + " s")
                    text_ouverture.config(text=text)
                else:
                    text = "Arrêt de la musique dans " + str(format(time_relase, '.1f') + " s")
                    text_ouverture.config(text=text)
        else:
            if music_is_playing:
                text = "Lecture en cours : " + song_current
                text_ouverture.config(text=text)
            else:
                text_ouverture.config(text="")
    old_hand_posture = hand_posture

# Fonction pour fermer la fenétre tkinter
def close_fenetre(e):
    fenetre.destroy()

# Bind la touche espace pour fermer la fenêtre
fenetre.bind('<Escape>', lambda e:close_fenetre(e))

# Fonction qui va être lancer en boucle
def video_stream():
    global old_hand_posture
    _, frame = cap.read() # Lecture de frame par frame du flux caméra
    imgFinal = frame.copy() # Copie de la frame
    hands, img = detector.findHands(frame, draw=True) # Detection des mains sur la frame

    # Si une main est détécter
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        imgWhite = np.ones((300, 300, 3), np.uint8) * 255 #Création d'une image blanche en 300x300
        imgCrop = img[y-20:y + h+20, x-20:x + w+20] # Crop l'image originale ou se trouve la main
        aspectRatio = h/w
        
        try:
            #Conversion pour avoir une image de la main resize en 300x300
            if aspectRatio > 1:
                k = 300/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, 300))
                wGap = math.ceil((300-wCal)/2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = 300/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (300, hCal))
                hGap = math.ceil((300-hCal)/2)
                imgWhite[hGap:hCal + hGap,:] = imgResize
            
            #Prediction de l'IA
            prediction = classifier.getPrediction(imgWhite)
            text_prediction.config(text="Prediction : " + str(prediction[1]))
            waiting(prediction[1])
        except:
            pass
    else:
        text_prediction.config(text="Prediction : Pas de main détecter")
        old_hand_posture = 6
        if music_is_playing:
                text = "Lecture en cours : " + song_current
                text_ouverture.config(text=text)
        else:
            text_ouverture.config(text="")
    
    


    cv2image = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    weebcam_label.imgtk = imgtk
    weebcam_label.configure(image=imgtk)
    weebcam_label.after(1, video_stream) 

video_stream()
fenetre.mainloop()