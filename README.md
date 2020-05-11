# yolo_post_processing
csc621 sfsu

The darknet and YOLO files were for running my YOLO tests. If we had gotten to use my part of the project, those would not have been included in my part.

The Image Segmentation folder and subsequent files were for my personal understanding of how image segmentation worked, to give me a bit more context. It was interesting stuff and I did put some work into it, but not really relevant to our project.

The process.py file is where my work for this project is. The post_processing() function takes a path to the original image, and the YOLO output data from that image; it then checks the YOLO confidence on each box, filtering out "bad" boxes, and then computes Intersection over Union as a part of Non-Max Suppression to remove any excess or duplicate boxes, and displays the final image.

I used the Python Pickle library to cache the YOLO test data that I was working with, making it quicker and easier to perform my tests.
