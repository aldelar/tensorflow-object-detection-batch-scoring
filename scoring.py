from azureml_user.parallel_run import EntryScript
from azureml.core.run import Run
from azureml.core import Model

import os,argparse
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
    global inference_batch_size
   
    logger = EntryScript().logger
    logger.info("==> scoring.py init()")

    parser = argparse.ArgumentParser()
    parser.add_argument('--images-scored-folder', type=str, dest='images_scored_folder', help='images scored folder')
    parser.add_argument('--model-name', type=str, dest='model_name', help='model name')
    parser.add_argument('--model-version', type=str, dest='model_version', help='model version')
    parser.add_argument('--inference-batch-size', type=int, dest='inference_batch_size', help='inference batch size',default=5)
    args, unknown_args = parser.parse_known_args()
    
    images_scored_folder = args.images_scored_folder
    model_name = args.model_name
    model_version = args.model_version
    inference_batch_size = args.batch_size
    print("images_scored_folder:", images_tiled_scored_folder)
    print("model_name:", model_name)
    print("model_version:", model_version)
    print("inference)batch_size:", inference_batch_size)

    run = Run.get_context()
    run.log('model_name',model_name)
    run.log('model_version',model_version)
    run.log('inference_batch_size',inference_batch_size)
    
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
    return load_model(model_path, custom_objects={'relu6': tf.nn.relu6,'tf': tf})

# process each batch of images
# NOTE: that here, we also have an 'inference batch size' *how many images do we send to the GPU at once), which can be independently controlled from the ParallelRunStep batch size. Obviously, the Step bacth size should be bigger than the inference batch size
def run(mini_batch):
    results = []
    images_paths = []
    current_batch_size = 0
    for image_path in mini_batch:
        image_name = os.path.basename(image_path)
        results.append(image_name) # metadata to just indicate which image we are processing
        # loading up the inference batch
        images_paths.append(image_path)
        current_batch_size = current_batch_size + 1
        if current_batch_size == inference_batch_size:
            # score batch
            images_scored = score_batch(images_paths)
    # score remaming images if any
    if current_batch_size > 0:
        score_batch(images_paths)
    return results

#
def score_batch(images_paths):
    print(f"       > scoring batch({images_paths})",flush=True)
    images = []
    for image_path in images_paths:
        images.append(np.expand_dims(Image.open(image_path), axis=0))
    images_scored = model.predict_on_batch(images)
    # save the outputs
'''    for i in range(len(images_scored)):
        output_array = np.squeeze(images_scored[i])
        im_out = Image.fromarray(output_mask)
        #im_out.save(image_tile_scored_file_paths[i],'PNG')
'''    
    image_paths.clear() # reset batch

#
def score_images(image_tile_file_paths,image_tile_scored_file_paths):
    image_tiles = []
    for image_tile_file_path in image_tile_file_paths:
        image_tiles.append(np.expand_dims(Image.open(image_tile_file_path), axis=0))
    image_tiles = np.squeeze(image_tiles,axis=1)
    image_tiles_scored = model.predict_on_batch(image_tiles)
