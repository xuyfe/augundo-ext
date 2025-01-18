import os, gdown
from zipfile import ZipFile


# URL to checkpoints
MONODEPTH2_RESNET18_MONO_640X192_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip'

MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip'

MONODEPTH2_RESNET18_STEREO_640X192_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip'

MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip'

MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip'

MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip'

MONODEPTH2_RESNET18_MONO_1024X320_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip'

MONODEPTH2_RESNET18_STEREO_1024X320_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip'

MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip'

MONODEPTH2_RESNET50_MONO_640X192_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_resnet50_640x192.zip'

MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL_URL = \
    'https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_resnet50_no_pt_640x192.zip'


MONODEPTH2_MODELS_DIRPATH = os.path.join('external_models', 'monocular', 'monodepth2')

# ResNet18 based models
MONODEPTH2_RESNET18_MODELS_DIRPATH = os.path.join(MONODEPTH2_MODELS_DIRPATH, 'resnet18')

# Monocular training at 640x192
MONODEPTH2_RESNET18_MONO_640X192_MODEL = 'mono_640x192'
MONODEPTH2_RESNET18_MONO_640X192_MODEL_FILENAME = MONODEPTH2_RESNET18_MONO_640X192_MODEL + '.zip'
MONODEPTH2_RESNET18_MONO_640X192_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_MONO_640X192_MODEL_FILENAME)

MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL = 'mono_nopt_640x192'
MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL_FILENAME = MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL + '.zip'
MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL_FILENAME)

# Stereo training at 640x192
MONODEPTH2_RESNET18_STEREO_640X192_MODEL = 'stereo_640x192'
MONODEPTH2_RESNET18_STEREO_640X192_MODEL_FILENAME = MONODEPTH2_RESNET18_STEREO_640X192_MODEL + '.zip'
MONODEPTH2_RESNET18_STEREO_640X192_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_STEREO_640X192_MODEL_FILENAME)

MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL = 'stereo_nopt_640x192'
MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL_FILENAME = MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL + '.zip'
MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL_FILENAME)

# Monocular + stereo training at 640x192
MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL = 'mono_stereo_640x192'
MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL_FILENAME = MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL + '.zip'
MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL_FILENAME)

MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL = 'mono_stereo_nopt_640x192'
MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL_FILENAME = MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL + '.zip'
MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL_FILENAME)

# Monocular training at 1024x320
MONODEPTH2_RESNET18_MONO_1024X320_MODEL = 'mono_1024x320'
MONODEPTH2_RESNET18_MONO_1024X320_MODEL_FILENAME = MONODEPTH2_RESNET18_MONO_1024X320_MODEL + '.zip'
MONODEPTH2_RESNET18_MONO_1024X320_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_MONO_1024X320_MODEL_FILENAME)

# Stereo training at 1024x320
MONODEPTH2_RESNET18_STEREO_1024X320_MODEL = 'stereo_1024x320'
MONODEPTH2_RESNET18_STEREO_1024X320_MODEL_FILENAME = MONODEPTH2_RESNET18_STEREO_1024X320_MODEL + '.zip'
MONODEPTH2_RESNET18_STEREO_1024X320_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_STEREO_1024X320_MODEL_FILENAME)

# Monocular + stereo training at 1024x320
MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL = 'mono_stereo_1024x320'
MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL_FILENAME = MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL + '.zip'
MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET18_MODELS_DIRPATH,
    MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL_FILENAME)

# ResNet50 based models
MONODEPTH2_RESNET50_MODELS_DIRPATH = os.path.join(MONODEPTH2_MODELS_DIRPATH, 'resnet50')

# Monocular training at 640x192
MONODEPTH2_RESNET50_MONO_640X192_MODEL = 'mono_640x192'
MONODEPTH2_RESNET50_MONO_640X192_MODEL_FILENAME = MONODEPTH2_RESNET50_MONO_640X192_MODEL + '.zip'
MONODEPTH2_RESNET50_MONO_640X192_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET50_MODELS_DIRPATH,
    MONODEPTH2_RESNET50_MONO_640X192_MODEL_FILENAME)

MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL = 'mono_nopt_640x192'
MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL_FILENAME = MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL + '.zip'
MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_RESNET50_MODELS_DIRPATH,
    MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL_FILENAME)

for dirpath in [MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET50_MODELS_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

'''
Download ResNet18 monocular, stereo, monocular + stereo
'''
# Download ResNet18 monocular model at 640x192 pretrained on ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_MONO_640X192_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 monocular 640x192 model with pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_MONO_640X192_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_MONO_640X192_MODEL_URL,
        MONODEPTH2_RESNET18_MONO_640X192_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 monocular 640x192 model with pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_MONO_640X192_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_MONO_640X192_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_MONO_640X192_MODEL))

# Download ResNet18 monocular model at 640x192 without pretraining on ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 monocular 640x192 model without pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL_URL,
        MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 monocular 640x192 model without pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_MONO_NOPT_640X192_MODEL))

# Download ResNet18 stereo model at 640x192 pretrained on ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_STEREO_640X192_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 stereo 640x192 model with pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_STEREO_640X192_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_STEREO_640X192_MODEL_URL,
        MONODEPTH2_RESNET18_STEREO_640X192_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 stereo 640x192 model with pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_STEREO_640X192_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_STEREO_640X192_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_STEREO_640X192_MODEL))

# Download ResNet18 stereo model at 640x192 without pretraining on ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 stereo 640x192 model without pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL_URL,
        MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 stereo 640x192 model without pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_STEREO_NOPT_640X192_MODEL))

# Download ResNet18 monocular + stereo model at 640x192 pretrained on ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 monocular + stereo 640x192 model with pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL_URL,
        MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 monocular + stereo 640x192 model with pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_MONO_STEREO_640X192_MODEL))

# Download ResNet18 monocular + stereo model at 640x192 without pretraining on ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 monocular + stereo 640x192 model without pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL_URL,
        MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 monocular + stereo 640x192 model without pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_MONO_STEREO_NOPT_640X192_MODEL))

# Download ResNet18 monocular model at 1024x320 pretrained on ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_MONO_1024X320_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 monocular 1024x320 model with pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_MONO_1024X320_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_MONO_1024X320_MODEL_URL,
        MONODEPTH2_RESNET18_MONO_1024X320_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 monocular 1024x320 model with pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_MONO_1024X320_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_MONO_1024X320_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_MONO_1024X320_MODEL))

# Download ResNet18 stereo model at 1024x320 pretrained on ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_STEREO_1024X320_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 stereo 1024x320 model with pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_STEREO_1024X320_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_STEREO_1024X320_MODEL_URL,
        MONODEPTH2_RESNET18_STEREO_1024X320_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 stereo 1024x320 model with pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_STEREO_1024X320_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_STEREO_1024X320_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_STEREO_1024X320_MODEL))

# Download ResNet18 monocular + stereo model at 1024x320 pretrained on ImageNet
if not os.path.exists(MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet18 monocular + stereo 1024x320 model with pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL_URL,
        MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet18 monocular + stereo 1024x320 model with pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET18_MODELS_DIRPATH, MONODEPTH2_RESNET18_MONO_STEREO_1024X320_MODEL))

# Download ResNet50 monocular with pretrained ImageNet
if not os.path.exists(MONODEPTH2_RESNET50_MONO_640X192_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet50 monocular 640x192 model with pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET50_MONO_640X192_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET50_MONO_640X192_MODEL_URL,
        MONODEPTH2_RESNET50_MONO_640X192_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet50 monocular 640x192 model with pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET50_MONO_640X192_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET50_MONO_640X192_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET50_MODELS_DIRPATH, MONODEPTH2_RESNET50_MONO_640X192_MODEL))

# Download ResNet50 monocular without pretrained ImageNet
if not os.path.exists(MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL_FILEPATH):
    print('Downloading monodepth2 ResNet50 monocular 640x192 model without pretrained ImageNet initialization to {}'.format(
        MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL_FILEPATH))
    gdown.download(
        MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL_URL,
        MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL_FILEPATH,
        quiet=False)
else:
    print('Found monodepth2 ResNet50 monocular 640x192 model without pretrained ImageNet initialization in {}'.format(
        MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(
        os.path.join(MONODEPTH2_RESNET50_MODELS_DIRPATH, MONODEPTH2_RESNET50_MONO_NOPT_640X192_MODEL))
