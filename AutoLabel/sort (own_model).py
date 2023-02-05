import os, sys, shutil

#Verification si le dataset est créer
try:
    labels = os.listdir("./own_dataset")
except:
    print("Veuillez d'abord créer votre propore dataset avant de lancer le script avec le dossier 'own_dataset'")
    sys.exit()

def sort_own_data():
    #Verification si un dossier traning_data est créer 
    try:
        os.listdir("./training_data")
    except:
        print("Dossier 'training_data' introuvable, création du dossier en cours...")
        os.mkdir("./training_data")
        for label in labels:
            os.mkdir("./training_data/" + label)

    #Verification si un dossier verfication_data est créer 
    try:
        os.listdir("./validation_data")
    except:
        print("Dossier 'validation_data' introuvable, création du dossier en cours...")
        os.mkdir("./validation_data")
        for label in labels:
            os.mkdir("./validation_data/" + label)

    for label in labels:
        data = os.listdir("./own_dataset/" + label)
        split = int(len(data) * 0.8)
        training_data = data[:split]
        validation_data = data[split:]
        for element in training_data:
            shutil.copy("./own_dataset/" + label + "/" + element, "./training_data/" + label + "/" + element)
        for element in validation_data:
            shutil.copy("./own_dataset/" + label + "/" + element, "./validation_data/" + label + "/" + element)

sort_own_data()