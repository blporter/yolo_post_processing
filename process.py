import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yolo as y
import tensorflow as tf
from keras import backend as K

DEBUG = True
MAKE_CACHE = True


def intersection_over_union(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = (x2 - x1) * (y2 - y1)

    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union = (box1_area + box2_area) - intersection

    return intersection / union


def compute_nms(boxes, scores):
    max_boxes_tensor = K.variable(10, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    # Get the list of indices corresponding to boxes to keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=0.5)

    # Select only nms_indices boxes and return them
    return K.gather(boxes, nms_indices)


def get_box_components(bounding_box):
    b_x = bounding_box[0]
    b_y = bounding_box[1]
    b_h = bounding_box[2]
    b_w = bounding_box[3]

    return b_x, b_y, b_h, b_w


def draw_box(bounding_box, axes):
    b_x, b_y, b_h, b_w = get_box_components(bounding_box)

    # Creates the bounding box for the given output on the image
    axes.add_patch(
        patches.Rectangle(
            (b_x - (b_h / 2), b_y - (b_w / 2)),
            b_h,
            b_w,
            linewidth=1,
            edgecolor='r',
            fill=0))


def post_processing(original_image, yolo_output):
    # Open the original image in matplotlib, given the path
    image = plt.imread(original_image)
    plt.imshow(image)

    # The axes and bounding boxes will be overlain on the original image
    axes = plt.gca()
    boxes_list = []
    scores_list = []

    for o in yolo_output:
        object_class = str(o[0], 'utf-8')
        confidence = o[1]
        bounding_box = o[2]

        # This confidence threshold can be changed depending on our data.
        if confidence > 0.8:
            b_x, b_y, b_h, b_w = get_box_components(bounding_box)
            boxes_list.append([b_x, b_y, b_x + b_w, b_y + b_h])
            scores_list.append(confidence)
            if DEBUG:
                print(f'{confidence * 100:.2f}% sure that this is a(n) {object_class}.')
                print(f'(x, y): ({b_x:.2f}, {b_y:.2f}); (w, h): ({b_w:.2f}, {b_h:.2f})')

    new_boxes_list = compute_nms(boxes_list, scores_list)

    for box in new_boxes_list:
        draw_box(box, axes)

    if DEBUG:
        plt.show()


if __name__ == "__main__":
    import pickle

    # In my tests, I needed to pass my image to yolo and pickle.
    image_path = 'darknet/data/dog.jpg'

    # I cached some test data. Set MAKE_CACHE to true to redo the cache.
    if MAKE_CACHE:
        y.run_yolo(image_path)

    # Loads the yolo data from the cache.
    with open('yolo_output.pickle', 'rb') as f:
        yolo_data = pickle.load(f)

    # Call my function given the original image path and the yolo data
    post_processing(image_path, yolo_data)
