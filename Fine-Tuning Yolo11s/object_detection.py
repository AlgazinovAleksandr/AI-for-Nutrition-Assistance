import ultralytics
import cv2
from ultralytics import YOLO
import supervision as sv
from constants import api_key
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as IPImage, display
import matplotlib.pyplot as plt

def detect_objects(model_yolo, image_path, name_images):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    results = model_yolo(img)

    # Get base filename (without extension)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"{base_filename}_{timestamp}"

    # Create labeled image
    labeled_img = pil_img.copy()
    draw = ImageDraw.Draw(labeled_img)
    font = ImageFont.load_default()
    # Process detection results
    detection_info = {}
    if len(results) > 0:
        boxes = results[0].boxes
        # Count detected classes
        class_counts = {}
        for i, box in enumerate(boxes):
            # Get bounding box and class
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = results[0].names[cls]

            # Update class count
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

            # Current object index
            obj_idx = class_counts[class_name]

            # Draw bounding box and label on the image
            color = (255, 0, 0)  # Red border
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"{class_name} {conf:.2f}"
            text_w, text_h = draw.textsize(label, font=font) if hasattr(draw, 'textsize') else (100, 20)
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w, y1], fill=color)
            draw.text((x1, y1 - text_h - 2), label, fill=(255, 255, 255), font=font)

        # Save detection information for return value
        detection_info = {
            "object_count": len(boxes),
            "class_counts": class_counts,
            "original_image": image_path
        }
    else:
        # No objects detected
        detection_info = {
            "object_count": 0,
            "class_counts": {},
            "original_image": image_path
        }

    # Save labeled image (always use the requested filename)
    labeled_path = f"output_images/{name_images}.png"
    labeled_img.save(labeled_path)
    detection_info["labeled_image"] = labeled_path

    print(f"Detection completed! Found {detection_info['object_count']} objects")
    print(f"Labeled image saved to: {labeled_path}")
    return detection_info

def detect_with_confidence(model, image_path, conf=0.1, save=False, file_name=None):
    image = cv2.imread(image_path)
    results = model(image, conf=0.1, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    sv.plot_image(annotated_image)
    if save:
        cv2.imwrite('output_images/file_name.png', annotated_image)

def plot_boxes(image_path, results):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for box in results[0].boxes:
        # Get box coordinates [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Get class and confidence
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = f"{results[0].names[cls_id]} {conf:.2f}"
        
        # Plot box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    plt.figure(figsize=(12,8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def print_labels(results, model):
    labels = [model.names[int(box.cls)] for box in results[0].boxes]
    print(', '.join(labels) if labels else "No detections")