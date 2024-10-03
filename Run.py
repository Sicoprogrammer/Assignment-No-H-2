# Create the run.py script
with open("run.py", "w") as f:
    f.write("""
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')


def apply_mask(image, boxes):
    for box in boxes:
        x_min, y_min, x_max, y_max = [int(coord) for coord in box]
        image[y_min:y_max, x_min:x_max] = [0, 0, 255]  # Red mask
    return image

# Helper function for rotating the segmented object
def rotate_object(image, box, angle):
    x_min, y_min, x_max, y_max = [int(coord) for coord in box]
    
  
    if x_min < 0 or y_min < 0 or x_max > image.shape[1] or y_max > image.shape[0]:
        print(f"Invalid box coordinates: {box}. Skipping rotation.")
        return image

    object_img = image[y_min:y_max, x_min:x_max]
    if object_img.size == 0:
        print(f"Invalid object image for rotation from box: {box}. Skipping rotation.")
        return image
    center = (object_img.shape[1] // 2, object_img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(object_img, rotation_matrix, (object_img.shape[1], object_img.shape[0]))

   
    image[y_min:y_max, x_min:x_max] = rotated
    return image

def main(args):
    # Load the image
    image = Image.open(args.image)
    image_np = np.array(image)

   
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Extract bounding boxes and labels
    boxes = outputs.logits.softmax(-1)[0, :, :-1].max(-1)[1]
    selected_boxes = outputs.pred_boxes[0][boxes == 1]  # Assume the first class is the object class

    # Convert boxes to [x_min, y_min, x_max, y_max] format
    boxes = selected_boxes.detach().cpu().numpy() * [image_np.shape[1], image_np.shape[0], image_np.shape[1], image_np.shape[0]]

    if args.azimuth or args.polar:
        # Task 2: Change pose
        angle = args.azimuth  # Use azimuth angle for rotation
        for box in boxes:
            image_np = rotate_object(image_np, box, angle)
    else:
        # Task 1: Apply red mask 
        image_np = apply_mask(image_np, boxes)

    # result
    cv2.imwrite(args.output, image_np)
    print(f"Output saved to {args.output}")


""")
