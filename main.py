import os

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


if __name__ == '__main__':
    # number of iterations to train for
    TRAIN_MAX_ITER = 10000

    # used to run external scripts
    PYTHON_PATH = 'C:\\Users\\user\\PycharmProjects\\imageProcessingDemo\\venv\\Scripts\\python.exe '

    # do not use GPUs on this machine
    tf.config.set_visible_devices([], 'GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    paths = {
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
        'IMAGE_PATH': os.path.join('Tensorflow', 'workspace', 'images'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
        'TFJS_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
        'TFLITE_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
        'PROTOC_PATH': os.path.join('Tensorflow', 'protoc')
    }

    # custom files, https://github.com/nicknochnack/TFODCourse/blob/main/2.%20Training%20and%20Detection.ipynb
    files = {
        'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

    # make all the folders, if they do not exist
    for path in paths.values():
        if not os.path.exists(path):
            if os.name == 'nt':
                os.mkdir(path)

    # assume windows machine, otherwise tools setup is different
    if os.name == 'nt':
        # installation verification, do once - avoid after
        # VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders',
        #                                    'model_builder_tf2_test.py')
        # os.system(VERIFICATION_SCRIPT)

        # we assume a single label, a single object type to detect
        labels = [{'name': 'apple', 'id': 1}]

        # write labels to file
        with open(files['LABELMAP'], 'w') as f:
            for label in labels:
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')

        # call generate TF records from https://github.com/nicknochnack/GenerateTFRecord
        os.system(PYTHON_PATH + files[
            'TF_RECORD_SCRIPT'] + ' -x ' + os.path.join(paths['IMAGE_PATH'], 'train') + ' -l ' + files[
                      'LABELMAP'] + ' -o ' + os.path.join(paths['ANNOTATION_PATH'], 'train.record'))
        os.system(PYTHON_PATH + files[
            'TF_RECORD_SCRIPT'] + ' -x ' + os.path.join(paths['IMAGE_PATH'], 'test') + ' -l ' + files[
                      'LABELMAP'] + ' -o ' + os.path.join(paths['ANNOTATION_PATH'], 'test.record'))

        # copy new pipeline config
        os.system('copy ' + os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME,
                                         'pipeline.config') + ' ' + os.path.join(paths['CHECKPOINT_PATH']))

        # create basic config
        config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

        # customize config pipeline
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)

        pipeline_config.model.ssd.num_classes = len(labels)
        pipeline_config.train_config.batch_size = 4
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'],
                                                                         PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
            os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
            os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

        # write new config to file
        config_text = text_format.MessageToString(pipeline_config)
        with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], 'wb') as f:
            f.write(config_text)

        # should check for GPUs
        # print(os.environ['CUDA_VISIBLE_DEVICES'])

        TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

        # RUN THE TRAINING, this takes some time
        command_train = PYTHON_PATH + " {} --model_dir={} --pipeline_config_path={} --num_train_steps={}".format(TRAINING_SCRIPT,
                                                                                                     paths[
                                                                                                         'CHECKPOINT_PATH'],
                                                                                                     files[
                                                                                                         'PIPELINE_CONFIG'],
                                                                                                     TRAIN_MAX_ITER)
        print(command_train)
        os.system(command_train)

        # OPTIONAL, RUN THE TRAINING - USEFUL TO EVALUATE DIFFERENT MODELS
        # command_test = PYTHON_PATH + " {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT,
        #                                                                                           paths[
        #                                                                                               'CHECKPOINT_PATH'],
        #                                                                                           files[
        #                                                                                               'PIPELINE_CONFIG'],
        #                                                                                           paths[
        #                                                                                               'CHECKPOINT_PATH'])
        # print(command_test)
        # os.system(command_test)
