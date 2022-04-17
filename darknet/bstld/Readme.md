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
