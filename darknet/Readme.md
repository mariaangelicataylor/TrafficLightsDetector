# Here you should clone and make darknet 


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
