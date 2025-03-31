
My First Project - v3 2025-03-30 6:39pm
==============================

This dataset was exported via roboflow.com on March 30, 2025 at 10:40 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 144 images.
Fo-ofoods are annotated in YOLOv12 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit (white edges))

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 21 percent of the image
* Random rotation of between -4 and +4 degrees
* Random exposure adjustment of between -12 and +12 percent
* Random Gaussian blur of between 0 and 2.8 pixels


