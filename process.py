import matplotlib.pyplot as plt
import matplotlib.patches as patches

DEBUG = True
MAKE_CACHE = False


def post_processing(original_image, yolo_output):
    axes = plt.gca()

    for o in yolo_output:
        object_class = str(o[0], 'utf-8')
        confidence = o[1]
        bounding_box = o[2]

        # This confidence threshold can be changed depending on our data.
        # TODO: add IoU and NMS
        if confidence > 0.8:
            b_x = bounding_box[0]
            b_y = bounding_box[1]
            b_h = bounding_box[2]
            b_w = bounding_box[3]

            # Creates the bounding box for the given output on the image
            axes.add_patch(
                patches.Rectangle(
                    (b_x - (b_h / 2), b_y - (b_w / 2)),
                    b_h,
                    b_w,
                    linewidth=1,
                    edgecolor='r',
                    fill=0))

            if DEBUG:
                print(f'{confidence * 100:.2f}% sure that this is a(n) {object_class}.')
                print(f'(x, y): ({b_x:.2f}, {b_y:.2f}); (w, h): ({b_w:.2f}, {b_h:.2f})')

    # TODO: I need this function to return the final image, with boxes.
    plt.show()


if __name__ == "__main__":
    import pickle

    # In my tests, I needed to pass my image to yolo and pickle.
    image_path = 'darknet/data/dog.jpg'

    # I cached some test data. Set MAKE_CACHE to true to redo the cache.
    if MAKE_CACHE:
        import yolo as y

        y.run_yolo(image_path)

    # Loads the yolo data from the cache.
    with open('yolo_output.pickle', 'rb') as f:
        yolo_data = pickle.load(f)

    # Open the image in matplotlib, given the path.
    image = plt.imread(image_path)
    plt.imshow(image)

    # Call my function given the original image and the yolo data
    post_processing(image, yolo_data)
