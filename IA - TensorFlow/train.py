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
import wget
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


#On initialise les modules de pre-entrainement
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
#On choisi le module d'entrainement que l'on va utiliser
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
#On affiche l'url afin de pourvoir l'installer par la suite
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
#On initialise 
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'


#On initialises des noms comptenants les chemins vers les dossiers qui nous serviront plus tard
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('./Tensorflow', 'workspace','annotations'),
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


for path in paths.values():
        if os.name == 'nt':
            os.mkdir (path)

#Si le fichier "apimodel_path" que nous allons utiliser n'est pas créé, on l'importe à l'aide d'un git clone
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    #!git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}
    pass


#On installe " Tensorflow Objet Detection "

#On telecharge protoc, qui est utiliser par " Tensorflow Objet Detection "
if os.name=='nt':
    url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url)
    #os.move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}
    #os.cd (paths['PROTOC_PATH']) && tar -xf protoc-3.15.6-win64.zip
    #os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
    #On installe la vrai version de " Tensorflow Objet Detection "
    #os.cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install
    #os.cd Tensorflow/models/research/slim && pip install -e . 


# On verifie si Tensorflow object detection est bien installé
VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
os.python (VERIFICATION_SCRIPT)


#On installe le modele d'entrainement et on va le placer dans le fichier créé pour plus haut
if os.name == 'nt':
    wget.download(PRETRAINED_MODEL_URL)
    #os.move (PRETRAINED_MODEL_NAME+'.tar.gz') (paths['PRETRAINED_MODEL_PATH'])
    #os.cd (paths['PRETRAINED_MODEL_PATH']) && tar -zxvf (PRETRAINED_MODEL_NAME+'.tar.gz')


#On créé les labels qui seront ce que l'ia devra trouver sur l'image
labels = [{'name':'zero', 'id':1}, {'name':'un', 'id':2}, {'name':'deux', 'id':3}, {'name':'trois', 'id':4}, {'name':'quatre', 'id':5}, {'name':'cinq', 'id':6}]

#On créé le programme python pour chaque labels
with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

#On clone un fichier venant du git de celui qui à fait la video
#Le fichier s'appelle TF Records et va permettre de convertir nos images et nos annotations en fichier que nous pourront utiliser
if not os.path.exists(files['TF_RECORD_SCRIPT']):
    #os.git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}


#On crée les modules train et test qui seront utiliser pour entrainer notre machine
#os.python (files['TF_RECORD_SCRIPT']) -x (os.path.join(paths['IMAGE_PATH'], 'train')) -l (files['LABELMAP']) -o (os.path.join(paths['ANNOTATION_PATH'], 'train.record')) 
#os.python (files['TF_RECORD_SCRIPT']) -x (os.path.join(paths['IMAGE_PATH'], 'test')) -l (files['LABELMAP']) -o (os.path.join(paths['ANNOTATION_PATH'], 'test.record'))


#On copie un model d'entrainement avec une configuration predefinie
#if os.name == 'nt':
    # os.copy (os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')) (os.path.join(paths['CHECKPOINT_PATH']))
    pass

#Dans toute cette partie, On modifie les lignes afin d'y intégrer les infos de notre ia comme le nombre de labels qu'on à rentrer
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  
pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   


# On initialise notre module d'entrainement
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])


# Dans un nouveau terminal on execute la commande suivante : " python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --num_train_steps=2000 "
#Notre IA s'entraine à reconnaitre ce que nous lui avons demandé avec les images que l'on a prise


#On initialise le module d'evaluation
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])

# Dans un nouveau terminal on execute la commande suivante : " python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet "
# On va évaluer notre IA afin qu'elle corige ses potentielles erreursa