# Detecting Traffic Recognition YOLOv3

Maria Angelica Taylor 
AI for Autonomous Vehicles WPI. 
Assigment 8

We’ve all been there: a light turns green and the car in front of you doesn’t budge. No one likes to get stuck behind a vehicle that doesn’t notice when a light change. Also, a system that can countdown on red light the time remaining until a change to green can save a significant quantity of fuel in city driving (e.g., restart engine five seconds before green) and advise driver to start braking early if it will not make it through a green light. 
In this project, we will develop a model to recognize traffic-light state in the car driving direction. We will use the bosh data set and explain step by step to make test this model. 
This project is a fork of https://github.com/berktepebag/Traffic-light-detection-with-YOLOv3-BOSCH-traffic-light-dataset

This repo contains the instructions to set up your enviroment and the folder structure, but you need to clone the darknet (YOLO) and bstld repos.

## Step 1: Download YOLOv3

Here is the <a href='https://pjreddie.com/darknet/yolo/' >YOLO offical page</a> to proceed with setup and for more details. Let's clone and make it with:

```html
git clone https://github.com/pjreddie/darknet
cd darknet
make
```

Now we need some example weights to run YOLO, download it from <a href="https://pjreddie.com/media/files/yolov3.weights">here</a> and save it into darknet folder.

Now we can run an example:

```html
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

A result image will appear and we can see that YOLO found a dog, a bicycle and a truck. YOLO can be used for multiple images, with webcam and videos.

## Step 2: Download bosh tools

As we already know that we need labels for the input images. In a standart CNN it would be a label for each image but since we are looking for parts of one image we need more than this. So YOLO is asking for a .txt file for each image as:
```html
<object-class> <x> <y> <width> <height>
```

Bosch Small Traffic Lights Dataset is coming with a Python script which turns the dataset into Pascal-VOC like dataset. It is good because YOLO has a script for converting VOC dataset to YOLO styled input. First clone the repository into the extracted dataset folder:  
```html
git clone git@github.com:bosch-ros-pkg/bstld.git
```

If you already don't have SSH key and getting an error, you have to set one and link it to your Github account in order to clone this repository. You can follow <a href="https://help.github.com/en/articles/connecting-to-github-with-ssh"> Github tutorial for SSH</a>.


## I cloned this repo (bstld) inside the darknet folder 

## Step 3: Download Bosch Small Traffic Lights Dataset:

https://hci.iwr.uni-heidelberg.de/node/6132

Register and a code will be sent to our e-mail address:

<img src="imgs/download_bosch_traffic_dataset.png" alt="Download Bosch Small Traffic Lights Dataset">

Dataset is around 6 GB, so it will take a while to download it. When download is done you should be using 7-zip to open it (In Ubuntu Archieve Manager is not opening the zipped file!), there are 5093 images for training.

## Step 4: Data Folders Preparation - bstld

## Inside the bstld folder you will create a folder named rgb, inside rbg you need to create a folder train and a folder test. Inside train you will crate three folders: 

1. traffic_light_images
2. traffic_light_xmls
3. traffic_light_labels

```html
mkdir rgb
cd rgb
mkdir train
mkdir test
cd train
mkdir traffic_light_images
mkdir traffic_light_xmls
mkdir traffic_light_labels
```
**Update:** PyYaml's load function has been <a href=https://stackoverflow.com/questions/69564817/typeerror-load-missing-1-required-positional-argument-loader-in-google-col>deprecated</a>, so if you are getting an error with yaml.load() you should change bosch_to_pascal.py line 60 to yaml.safe_load() .


Now go back to bstld folder and run bosch_to_pascal.py script, which will create necessary xml files for training with YOLO. Where first argument is PATH_TO_DATASET/train.yaml and second argument is rgb/train/traffic_light_xmls folder which we recently created:
```html
cd ../..
python bosch_to_pascal.py train.yaml rgb/train/traffic_light_xmls/
```

Now we have 5093 xml label files but we have to convert VOC to YOLO type labels with the script from darknet. 

## Step 5: traffic-lights folder in darknet
Let's go back to the darknet folder and create a folder named traffic-lights. We will put our files in this folder to reach them easily.

```html
mkdir traffic-lights 
```

#### Step 6: VOC -> YOLO
From darknet/scripts folder, make a copy of the voc_label.py and name it bosch_voc_to_yolo_converter.py and put it under traffic-lights folder. This script will convert VOC type labels to YOLO type labels.

```html
cp scripts/voc_label.py traffic-lights/bosch_voc_to_yolo_converter.py
```

Here we have to change classes names with our class names from the dataset.

```Python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys

sets=['traffic_lights']

classes = ["RedLeft", "Red", "RedRight", "GreenLeft", "Green", "GreenRight", "Yellow", "off"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xml_path_input,file_folder,file_name):
    in_file = open('%s'%(xml_path_input))
    out_file = open('%s/%s.txt'%(file_folder,file_name), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

output_folder = str(sys.argv[1])
xmls_list = str(sys.argv[2])
images_folder = str(sys.argv[3])

for image_set in sets:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    xml_paths = open(xmls_list).read().strip().split()
    list_file = open('%s.txt'%(image_set), 'w')
    for xml_path in xml_paths:
        #print("xml path: ",xml_path)
        xml_name = xml_path.split('/')[-1]
        #print("xml name:",xml_name)
        image_name = xml_name.split('.')[0]
        #print("image name: ",image_name)
        #print(images_folder+'/%s.png\n'%(image_name))
        list_file.write(images_folder+'%s.png\n'%(image_name))
        convert_annotation(xml_path,output_folder,image_name)
    list_file.close()
 ```

And for the arguments, we have to give:
1. output_folder for .txt files (PATH_TO_DATASET/rgb/train/traffic_light_labels)
2. xmls_list which is a .txt file that has the paths to the xml files and (we will create next)
3. images folder path which we are going to use for training. (PATH_TO_DATASET/rgb/train/traffic_light_images)

We need the paths of the .xml files as a list in a .txt file, in order to get it we will write a little Python script:

```html
cd traffic-lights
subl make_xml_list.py
```

```Python
import os
import sys

xmls_path = sys.argv[1] #xml files path

xml_files = []

#r = root, d = directories, f = xml_files

for r,d,f in os.walk(xmls_path):
	for file in f:
		if '.xml' in file:
			xml_files.append(os.path.join(r, file)) #Gets the whole file xmls_path
			#xml_files.append(os.path.splitext(file)[0]) # Gets only the name of the file without extension,path etc.	

file_num = len(xml_files)
print("Length of the .xml xml_files: ", file_num)

if not open('bosch_traffic_light_xmls_list.txt','w'):
	os.makefile('bosch_traffic_light_xmls_list.txt')

labels = open('bosch_traffic_light_xmls_list.txt','w')

for xml in xml_files:
	labels.write(xml + '\n')

labels.close()

#for f in xml_files:
	#print(f)
```

Save and run it:

```html
python make_xml_list.py PATH_TO_DATASET/rgb/train/traffic_light_xmls/
```
It will create bosch_traffic_light_xmls_list.txt file.

Let's copy the data/voc.names to traffic-lights and name it voc-bosch.names:

```html
cp ../data/voc.names voc-bosch.names
subl voc-bosch.names
```

and replace the items with:

1. RedLeft
2. Red
3. RedRight
4. GreenLeft
5. Green
6. GreenRight
7. Yellow
8. off


Now we can convert VOC to YOLO format:

We will use the folder PATH_TO_DATASET/rgb/train/traffic_light_labels for outputs, 
recently created bosch_traffic_light_xmls_list.txt and 
PATH_TO_DATASET/rgb/train/traffic_light_images for training images.

```html
python bosch_voc_to_yolo_converter.py ~/Datasets/Bosch-Traffic-Light-Dataset/rgb/train/traffic_light_labels/ bosch_traffic_light_xmls_list.txt ~/Datasets/Bosch-Traffic-Light-Dataset/rgb/train/traffic_light_images/
```

We have to create train.txt and test.txt which are list of the paths' of the relative images. Write a basic splitter script named train_test_split.py:

```html
subl train_test_split.py
```

```Python
import glob, os
import numpy as np
from sklearn.model_selection import train_test_split
import sys

image_paths_file = sys.argv[1] #traffic_lights/traffic_lights.txt

# Percentage of images to be used for the test set (float between 0-1)
percentage_test = float(sys.argv[2]);

img_paths = []
img_paths = open(image_paths_file).read().strip().split()

X_train, X_test= train_test_split(img_paths, test_size=percentage_test, random_state=31)

with open('train.txt', 'a') as train_file:
	for train in X_train:
		train_file.write(train + '\n')

with open('test.txt', 'a') as test_file:
	for test in X_test:
		test_file.write(test + '\n')
```

It takes recently created traffic_lights.txt file as first argument and second argument split percentage between 0 to 1. 

```html
python train_test_split.py traffic_lights.txt 0.2
```
test.txt and train.txt is created.

Create a backup folder inside traffic-lights folder where we will save our weights as we train:

```html
mkdir backup
```

Make a copy of the cfg/voc.data and name it voc-bosch.data .

```html
cp ../cfg/voc.data voc-bosch.data
```
Open it:

```html
classes= 20
train  = /home/pjreddie/data/voc/train.txt
valid  = /home/pjreddie/data/voc/2007_test.txt
names = data/voc.names
backup = backup
```

classes shows the number of the labels we would like to classify. From the dataset we can see that main lights are RedLeft, Red, RedRight, GreenLeft, Green, GreenRight, Yellow and off. Feel free to add or extract the ones you like. So our classes will be '8'. train.txt and test.txt are the text files which has the paths of the image files. names, are labels' names and as mentioned before we should get them from the database. Let's start updating the voc-bosch.data:

```html
classes= 8
train  = traffic-lights/train.txt
valid  = traffic-lights/test.txt
names = traffic-lights/voc-bosch.names
backup = traffic-lights/backup
```

Now we need one more thing to do to start training. Copy yolov3-tiny.cfg from darknet/cfg folder into traffic-lights folder and name it yolov3-tiny-bosch.cfg .

```html
cp ../cfg/yolov3-tiny.cfg yolov3-tiny-bosch.cfg
```

Open it:

You will see line 5,6,7 is commented out, uncomment them:

```html
Training
batch=64
subdivisions=2
```

<a href="https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects">Calculate number of filters: </a>

```html
filters= 3 x (5 + #ofclasses)
```

filters = 3 x (5+8) = 39

Change filters' size before '[yolo]' parameters (lines 127 and 171) with 39 and classes to 8 in '[yolo]' parameters (lines 135 and 177).

We will use the technique called transfer learning where we use the pre-trained VOC data and just change the end of the deep-neural-network.

Download <a href="https://pjreddie.com/media/files/darknet53.conv.74"> weights of the darknet53 model</a> and train:

```html
cd ..
./darknet detector train traffic-lights/voc-bosch.data traffic-lights/yolov3-tiny-bosch.cfg darknet53.conv.74
```

After training done (after 30000 images results are getting sufficient) try it with:
```html
./darknet detector demo traffic-lights/voc-bosch.data traffic-lights/yolov3-tiny-bosch.cfg traffic-lights/backup/yolov3-tiny-bosch_40000.weights <video file>
```

## Video Demo

Let's try our classifier with a video:

```html
./darknet detector demo traffic-lights/voc-bosch.data traffic-lights/yolov3-tiny-bosch.cfg traffic-lights//backup/yolov3-tiny-bosch_40000.weights <video file>
```

<img src="imgs/traffic_light_detector_demo.png" alt="YOLOv3 Traffic light detector demo">
	
### Probable Problems with OpenCv
	
If you are having problem while running make with OpenCv=1:
	
1. <a href="https://stackoverflow.com/questions/55306007/how-to-compile-yolov3-with-opencv">Modify Makefile</a>
Change opencv -> opencv4
- LDFLAGS+= `pkg-config --libs opencv4` -lstdc++
- COMMON+= `pkg-config --cflags opencv4` 
2. <a href="https://stackoverflow.com/questions/64885148/error-iplimage-does-not-name-a-type-when-trying-to-build-darknet-with-opencv"> Modify /src/image_opencv.cpp</a>
	
Add:
- #include "opencv2/core/core_c.h"
- #include "opencv2/videoio/legacy/constants_c.h"
- #include "opencv2/highgui/highgui_c.h"
	
Change:
- IplImage ipl = m -> IplImage ipl = cvIplImage(m);