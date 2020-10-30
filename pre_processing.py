from azureml_user.parallel_run import EntryScript
from azureml.core.run import Run

import os,argparse

from PIL import ImageFile, Image

# ===============
# ParallelRunStep: this script needs to implement a simple API that consists of two functions: init() and run()
# ===============

# init(): this code is executed once when the process starts up
def init():
    
    global logger
    global images_pre_processed_folder
   
    logger = EntryScript().logger
    logger.info("==> pre_processing.py init()")

    parser = argparse.ArgumentParser()
    parser.add_argument('--images-pre-processed-folder', type=str, dest='images_pre_processed_folder', help='images pre processed folder')
    args, unknown_args = parser.parse_known_args()
    images_pre_processed_folder = args.images_pre_processed_folder
    print("images_pre_processed_folder:", images_pre_processed_folder) # this is where the output of this step will be stored, at this stage to this code, it is just a system path you can write into
    
# run(): called multiple times until we run out of batches to process
def run(mini_batch):
    results = [] # this is metadata you can generate, one item per source item max, it will be stored/merged accross all processes into the file defined by the 'append_row_file_name' parameter of the pipeline step
    for image_path in mini_batch:
        # simple pre-processing: we just try to read the file as an image to see if it is an image
        try:
            image_name = os.path.basename(image_path)
            image = Image.open(image_path)
            image.save(os.path.join(images_pre_processed_folder,image_name),'PNG')
            # success, we flag it as valid in the generated metadata, and save it to our output for the next step
            results.append("valid,"+image_name)
        except Exception as e:
            print(e)
            # error: this file is not valid, we mark it as such in the metadata and do not move the image to the output of the step
            results.append("invalid,"+image_name)
    return results