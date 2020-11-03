from azureml_user.parallel_run import EntryScript
from azureml.core.run import Run
from azureml.core import Model

import os,argparse,json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import ImageFile, Image

#
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# import any custom packages copied to the target in the 'source_directory' setup
import scoring_custom_package

# read script params and load model from the Azure ML registry based on these parameters
def init():
    
    global logger
    global model
    global images_scored_folder
   
    logger = EntryScript().logger
    logger.info("==> scoring.py init()")

    parser = argparse.ArgumentParser()
    parser.add_argument('--images-scored-folder', type=str, dest='images_scored_folder', help='images scored folder')
    parser.add_argument('--model-name', type=str, dest='model_name', help='model name')
    parser.add_argument('--model-version', type=str, dest='model_version', help='model version')
    args, unknown_args = parser.parse_known_args()
    
    images_scored_folder = args.images_scored_folder
    model_name = args.model_name
    model_version = args.model_version
    print("images_scored_folder:", images_scored_folder)
    print("model_name:", model_name)
    print("model_version:", model_version)

    run = Run.get_context()
    run.log('model_name',model_name)
    run.log('model_version',model_version)
    
    # load model from AzureML registry
    model = load_model_from_registry(model_name,model_version)
    m = f"==> scoring.py init() -> model loaded: {model}"
    logger.info(m)

# load model from AzureML registry
def load_model_from_registry(model_name,model_version):
    m = f"    > load_model_from_registry({model_name},{model_version}) ..."
    logger.info(m)
    workspace = Run.get_context().experiment.workspace
    aml_model = Model(workspace, name=model_name, version=model_version)
    model_path = aml_model.download(target_dir='./outputs',exist_ok=True)
    return init_load(model_path)

# process each batch of images
def run(mini_batch):
    results = []
    for image_path in mini_batch:
        image_name = os.path.basename(image_path)
        image_name_base, image_name_extension = os.path.splitext(image_name)
        if image_name_extension != '.csv':
            detection = run_detect(image_path)
            f = open(os.path.join(images_scored_folder,image_name_base+'.json'),"w")
            f.write(json.dumps(detection))
            f.close()
            results.append(image_name) # metadata to just indicate which image we are processing
    return results

#
# insert your custom model load code here
def init_load(model_path):
    return load_model(model_path, custom_objects={'relu6': tf.nn.relu6,'tf': tf})

# insert your custom detection code here
def run_detect(image_path):
    print(f"       > detect({image_path})",flush=True)
    # insert detection code here and return detection results
    return { "detection": "None" }
