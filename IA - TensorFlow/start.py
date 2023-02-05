#On importe les librairies
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import os
import object_detection
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import keyboard
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import wget



#On initialise les modules de pre-entrainement
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
#On choisi le module d'entrainement que l'on va utiliser
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
#On affiche l'url afin de pourvoir l'installer par la suite
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'


#On initialises des noms comptenants les chemins vers les dossiers qui nous serviront plus tard
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }



files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}
    
#On créé les labels qui seront ce que l'ia devra trouver sur l'image
labels = [{'name':'zero', 'id':1}, {'name':'un', 'id':2}, {'name':'deux', 'id':3}, {'name':'trois', 'id':4}, {'name':'quatre', 'id':5}, {'name':'cinq', 'id':6}]

#On créé le programme python pour chaque labels
with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')



#On crée pipeline config et on créé le modele de détéction
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# On appelle les checkpoints créés lors du programme train et test
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

#On créé un index de catégorie à partir d'un fichier de labelmap
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


#On utilise la caméra numéro 0 de notre pc
cap = cv2.VideoCapture(0)
#On defini les dimension de l'image que l'on va capturer à partir de la caméra qui va apparaitre grace à cv2
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened(): 
    #On capture les frames de la vidéo enregistrée
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    #On crée un tenseur à partir d'une frame
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    #On detecte si l'on voit un label
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    #On converti les classes de detection en nombre entier
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    #On initialise label_id_offset
    label_id_offset = 1
    #On créé une copie de l'image
    image_np_with_detections = image_np.copy()


    #On affiche le cadrage affichant le label detécté
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                #On affiche la detection que si l'on est sur à 70% ou + que l'image correspond à un label
                min_score_thresh=.7,
                agnostic_mode=False)

    #On deffinie les dimensions de l'onglet
    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (1280, 720)))
    
    #Si on appuie sur q le programme s'arrête
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break