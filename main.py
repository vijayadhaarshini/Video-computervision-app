import streamlit as st
from streamlit_elements import elements,mui
from moviepy.editor import VideoFileClip
import cv2


def avi_to_mp4(input_path, output_path):
  """Converts an AVI file to MP4 format.

  Args:
      input_path: Path to the input AVI file.
      output_path: Path to save the converted MP4 file.
  """
  try:
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, fps=clip.fps)  # Preserve original frame rate
    print(f"Converted AVI to MP4: {input_path} -> {output_path}")
  except Exception as e:
    print(f"Error converting AVI: {e}")


st.title("Computer Vision Streamlit")
st.subheader("Video")
video = st.file_uploader("Upload a video file:", type=["mp4"])


if video is not None:
   st.video(video, format="video/mp4")
   file = st.text_input("Enter path")   


   from streamlit_pills import pills
   selected = pills("Computer Vision tasks", ["Object detection", "Instance segmentation","Video classification","Object tracking", "Pose recognition","Semantic segmentation"],index=None)
   st.write(selected)
   
   if selected == "Instance segmentation":
   
   

      from collections import defaultdict
      import cv2
      from ultralytics import YOLO
      from ultralytics.utils.plotting import Annotator, colors
# Dictionary to store tracking history with default empty lists
      track_history = defaultdict(lambda: [])
# Load the YOLO model with segmentation capabilities
      @st.cache_resource
      def model1():
         model1 = YOLO("yolov8n-seg.pt")
         return model1
   
      model=model1()
# Open the video file
      with st.spinner(":) Getting video "):
             if video is not None:
              cap = cv2.VideoCapture(file)
   
   
# Retrieve video properties: width, height, and frames per second
      w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
   
# Initialize video writer to save the output video with the specified properties
      out = cv2.VideoWriter("instance-segmentation-object-tracking.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w,h))
      while True:
    # Read a frame from the video
    
             ret, im0 = cap.read()
             if not ret:
                 print("Video frame is empty or video processing has been successfully completed.")
                 break

    # Create an annotator object to draw on the frame
             annotator = Annotator(im0, line_width=2)

    # Perform object tracking on the current frame
             results = model.track(im0, persist=True)

    # Check if tracking IDs and masks are present in the results
             if results[0].boxes.id is not None and results[0].masks is not None:
        # Extract masks and tracking IDs
                 masks = results[0].masks.xy
                 track_ids = results[0].boxes.id.int().cpu().tolist()

        # Annotate each mask with its corresponding tracking ID and color
                 for mask, track_id in zip(masks, track_ids):
                     annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=str(track_id))

    # Write the annotated frame to the output video
             out.write(im0)
    # Display the annotated frame
             cv2.imshow("instance-segmentation-object-tracking", im0)
       
    # Exit the loop if 'q' is pressed
             if cv2.waitKey(1) & 0xFF == ord("q"):
                 break

# Release the video writer and capture objects, and close all OpenCV windows
      out.release()
      cap.release()
      cv2.destroyAllWindows()
      avi_to_mp4("instance-segmentation-object-tracking.avi","instance.mp4")
      st.video("instance.mp4")
   

   if selected == "Pose recognition":
      import cv2
      import matplotlib.pyplot as plt
      net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
      inWidth = 368
      inHeight = 368
      thr= 0.2
      BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

      POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
      cap = cv2.VideoCapture(file)
      cap.set(3,800)
      cap.set(4,800)
      w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
      hello = cv2.VideoWriter("pose.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w,h))
      with st.spinner("🤖 AI is at Work! "):
       while cv2.waitKey(1) < 0:
          hasFrame,frame=cap.read()
          if not hasFrame:
              cv2.waitKey()
              break
          frameWidth=frame.shape[1]
          frameHeight=frame.shape[0]
          net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
          out = net.forward()
          out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
          assert(len(BODY_PARTS) == out.shape[1])
          points = []
          for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
              heatMap = out[0, i, :, :]
              _,conf,_,pt = cv2.minMaxLoc(heatMap)
              x=frameWidth*pt[0]/out.shape[3]
              y=frameHeight*pt[1]/out.shape[2]
              points.append((int(x), int(y)) if conf > thr else None)

          for pair in POSE_PAIRS:
              partFrom = pair[0]
              partTo = pair[1]
              assert(partFrom in BODY_PARTS)
              assert(partTo in BODY_PARTS)

              idFrom = BODY_PARTS[partFrom]
              idTo = BODY_PARTS[partTo]
              idFrom = BODY_PARTS[partFrom]
              idTo = BODY_PARTS[partTo]

              if points[idFrom] and points[idTo]:
                  cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                  cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                  cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

          t, _ = net.getPerfProfile()
          freq = cv2.getTickFrequency() / 1000
          cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
          cv2.imshow('hi',frame)
          hello.write(frame)
       hello.release()
       cap.release()
       cv2.destroyAllWindows()
       avi_to_mp4("pose.avi","pose.mp4")
       st.video("pose.mp4")
       
   
   


   if selected == "Object detection":   
      import cv2
      import numpy as np
      from PIL import Image, ImageDraw
   
# Load YOLOv3 model
      yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class labels
      yolo_classes = []
      with open('coco.names', 'r') as f:
          yolo_classes = f.read().splitlines()

# Function to perform YOLO object detection on a single image
      def perform_yolo_detection(img):
          height, width, _ = img.shape  # Get image dimensions

    # Preprocess image for YOLO
          blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
          yolo_net.setInput(blob)

    # Get YOLO output
          output_layers_names = yolo_net.getUnconnectedOutLayersNames()
          layer_outputs = yolo_net.forward(output_layers_names)

    # Parse YOLO output
          boxes = []
          confidences = []
          class_ids = []

          for output in layer_outputs:
              for detection in output:
                  scores = detection[5:]
                  class_id = np.argmax(scores)
                  confidence = scores[class_id]

                  if confidence > 0.7:
                      center_x = int(detection[0] * width)
                      center_y = int(detection[1] * height)
                      w = int(detection[2] * width)
                      h = int(detection[3] * height)

                      x = int(center_x - w/2)
                      y = int(center_y - h/2)

                      boxes.append([x, y, w, h])
                      confidences.append(float(confidence))
                      class_ids.append(class_id)

    # Apply non-maximum suppression
          indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

           
    # Draw bounding boxes and labels on the image
          font = cv2.FONT_HERSHEY_SIMPLEX  # Change font type
          thickness = 3  # Thickness of the rectangle border
          colors = np.random.uniform(0, 255, size=(len(boxes), 3))

          for i in indexes.flatten():
              x, y, w, h = boxes[i]
              label = str(yolo_classes[class_ids[i]])
              confidence = str(round(confidences[i], 2))
              color = tuple(int(c) for c in colors[i])

        # Draw outer rectangle using OpenCV (unchanged)
              cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)

        # Calculate text size for dynamic positioning (unchanged)
              (label_width, label_height), baseline = cv2.getTextSize(label + " " + confidence, font, 0.5, 2)

        # Draw inner rectangle using PIL for rounded corners
              pil_img = Image.fromarray(img)  # Convert OpenCV image to PIL image
              draw = ImageDraw.Draw(pil_img)
              draw.rounded_rectangle((x + 3, y - 20, x + label_width + 3, y + 2), radius=5, fill=color, outline=color, width=3)
              img = np.array(pil_img)  # Convert back to OpenCV image

        # Draw text using OpenCV (unchanged)
              cv2.putText(img, label + " " + confidence, (x, y - 5), font, 0.5, (255, 255, 255), 1)
          return img

# Open a live camera feed
      cap = cv2.VideoCapture(file)  # 0 indicates the default camera, you can change it if you have multiple cameras
      w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
      hello = cv2.VideoWriter("detection.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w,h))
      while True:
          ret, frame = cap.read()
          if not ret:
              break

    # Perform YOLO object detection on the frame
          yolo_detected_frame = perform_yolo_detection(frame)

    # Display the image with bounding boxes
          cv2.imshow('YOLO Object Detection', yolo_detected_frame)
          hello.write(yolo_detected_frame)
    
    
          if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to exit the loop
              break

      cap.release()
      cv2.destroyAllWindows()
      hello.release()
      avi_to_mp4("detection.avi","detection.mp4")
      st.video("detection.mp4")


   if selected == "Object tracking":
      from collections import defaultdict

      import cv2
      import numpy as np

      from ultralytics import YOLO

# Load the YOLOv8 model
      model = YOLO("yolov8n.pt")

# Open the video file
      
      cap = cv2.VideoCapture(file)
      w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
      hello = cv2.VideoWriter("tracking.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w,h))

# Store the track history
      track_history = defaultdict(lambda: [])

# Loop through the video frames
      while cap.isOpened():
    # Read a frame from the video
          success, frame = cap.read()

          if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
              results = model.track(frame, persist=True)

        # Get the boxes and track IDs
              boxes = results[0].boxes.xywh.cpu()
              track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
              annotated_frame = results[0].plot()

        # Plot the tracks
              for box, track_id in zip(boxes, track_ids):
                  x, y, w, h = box
                  track = track_history[track_id]
                  track.append((float(x), float(y)))  # x, y center point
                  if len(track) > 30:  # retain 90 tracks for 90 frames
                      track.pop(0)

            # Draw the tracking lines
                  points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                  cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
              cv2.imshow("YOLOv8 Tracking", annotated_frame)
              hello.write(annotated_frame)
 
        # Break the loop if 'q' is pressed
              if cv2.waitKey(1) & 0xFF == ord("q"):
                  break
          else:
        # Break the loop if the end of the video is reached
              break

# Release the video capture object and close the display window
      cap.release()
      cv2.destroyAllWindows()
      hello.release()
      avi_to_mp4("tracking.avi","tracking.mp4")
      st.video("tracking.mp4")
   if selected == 'Video classification':
      import av
      import torch
      import numpy as np

      from transformers import AutoImageProcessor, VideoMAEForVideoClassification
      from huggingface_hub import hf_hub_download

      np.random.seed(0)


      def read_video_pyav(container, indices):
          frames = []
          container.seek(0)
          start_index = indices[0]
          end_index = indices[-1]
          for i, frame in enumerate(container.decode(video=0)):
              if i > end_index:
               break
              if i >= start_index and i in indices:
               frames.append(frame)
          return np.stack([x.to_ndarray(format="rgb24") for x in frames])


      def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
   
          converted_len = int(clip_len * frame_sample_rate)
          end_idx = np.random.randint(converted_len, seg_len)
          start_idx = end_idx - converted_len
          indices = np.linspace(start_idx, end_idx, num=clip_len)
          indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
          return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
      
      container = av.open(file)

# sample 16 frames
      indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
      video = read_video_pyav(container, indices)

      image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
      model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

      inputs = image_processor(list(video), return_tensors="pt")

      with torch.no_grad():
          outputs = model(**inputs)
          logits = outputs.logits

# model predicts one of the 400 Kinetics-400 classes
      predicted_label = logits.argmax(-1).item()
      print(model.config.id2label[predicted_label])
      st.write(model.config.id2label[predicted_label])


   if selected == "Semantic segmentation":
      import os
      import numpy as np
      import cv2
      import zipfile
      import requests
      import glob as glob
      import tensorflow as tf
      import tensorflow_hub as hub
      import matplotlib.pyplot as plt
      from matplotlib.patches import Rectangle
      import warnings
      import logging
      import absl
      warnings.filterwarnings("ignore", module="absl")
      logging.captureWarnings(True)
      absl_logger = logging.getLogger("absl")
      absl_logger.setLevel(logging.ERROR)
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
      

      def load_image(path):
 
          image = cv2.imread(path)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image = np.expand_dims(image, axis=0)/255.0
          return image
      
      def class_to_rgb(mask_class, class_index):
          r_map = np.zeros_like(mask_class).astype(np.uint8)
          g_map = np.zeros_like(mask_class).astype(np.uint8)
          b_map = np.zeros_like(mask_class).astype(np.uint8)
          for class_id in range(len(class_index)):
              index = mask_class == class_id
              r_map[index] = class_index[class_id][0][0]
              g_map[index] = class_index[class_id][0][1]
              b_map[index] = class_index[class_id][0][2]
          seg_map_rgb = np.stack([r_map, g_map, b_map], axis=2)
          return seg_map_rgb
      def image_overlay(image, seg_map_rgb):
     
          alpha = 1.0 # Transparency for the original image.
          beta  = 0.6 # Transparency for the segmentation map.
          gamma = 0.0 # Scalar added to each sum.
          image = (image*255.0).astype(np.uint8)
          seg_map_rgb = cv2.cvtColor(seg_map_rgb, cv2.COLOR_RGB2BGR)
          image = cv2.addWeighted(image, alpha, seg_map_rgb, beta, gamma)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          return image
      def run_inference(img, model):
     
        pred_mask = model.predict(img).numpy()
        pred_mask = pred_mask[:,:,:,1:]
        pred_mask = np.squeeze(pred_mask)
        pred_mask_class = np.argmax(pred_mask, axis=-1)
        pred_mask_rgb = class_to_rgb(pred_mask_class, class_index)
             
        # Display the predicted color segmentation mask. 
        
 
        # Display the predicted color segmentation mask overlayed on the original image.
        overlayed_image = image_overlay(img[0], pred_mask_rgb)
        
        # Save the predicted mask as an image
        cv2.imwrite("predicted_mask.jpg", pred_mask_rgb)
        out.write(pred_mask_rgb)

# Display the saved image in Streamlit
        
        cv2.imwrite("over.jpg", overlayed_image)
        out1.write(overlayed_image)

# Display the saved image in Streamlit
        

      def plot_color_legend(class_index):
     
    # Extract colors and labels from class_index dictionary.
         color_array = np.array([[v[0][0], v[0][1], v[0][2]] for v in class_index.values()]).astype(np.uint8)
         class_labels = [val[1] for val in class_index.values()]    
    
              
    # Display color legend.
         for i, axis in enumerate(ax.flat):
 
             axis.imshow(color_array[i][None, None, :])
             axis.set_title(class_labels[i], fontsize = 8)
             axis.axis('off')

      model_url =  'https://tfhub.dev/google/HRNet/camvid-hrnetv2-w48/1'
      print('loading model: ', model_url)
 
      seg_model = hub.load(model_url)
      print('\nmodel loaded!')
      

      #/////////////////////////
      cap = cv2.VideoCapture(file)
      w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
   
# Initialize video writer to save the output video with the specified properties
      out = cv2.VideoWriter("sem.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w,h))
      out1 = cv2.VideoWriter("sem_2.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w,h))
      while True:
    # Read a frame from the video
    
        ret, im0 = cap.read()
        if not ret:
                 print("Video frame is empty or video processing has been successfully completed.")
                 break

        cv2.imwrite("extracted.jpg",im0)
      
        image = load_image("extracted.jpg")
        class_index = \
           {
         0: [(64, 128, 64),  'Animal'],
         1: [(192, 0, 128),  'Archway'],
         2: [(0, 128, 192),  'Bicyclist'],
         3: [(0, 128, 64),   'Bridge'],
         4: [(128, 0, 0),    'Building'],
         5: [(64, 0, 128),   'Car'],
         6: [(64, 0, 192),   'Cart/Luggage/Pram'],
         7: [(192, 128, 64), 'Child'],
         8: [(192, 192, 128),'Column Pole'],
         9: [(64, 64, 128),  'Fence'],
        10: [(128, 0, 192),  'LaneMkgs Driv'],
        11: [(192, 0, 64),   'LaneMkgs NonDriv'],
        12: [(128, 128, 64), 'Misc Text'],
        13: [(192, 0, 192),  'Motorcycle/Scooter'],
        14: [(128, 64, 64),  'Other Moving'],
        15: [(64, 192, 128), 'Parking Block'],
        16: [(64, 64, 0),    'Pedestrian'],
        17: [(128, 64, 128), 'Road'],
        18: [(128, 128, 192),'Road Shoulder'],
        19: [(0, 0, 192),    'Sidewalk'],
        20: [(192, 128, 128),'Sign Symbol'],
        21: [(128, 128, 128),'Sky'],
        22: [(64, 128, 192), 'SUV/Pickup/Truck'],
        23: [(0, 0, 64),     'Traffic Cone'],
        24: [(0, 64, 64),    'Traffic Light'],
        25: [(192, 64, 128), 'Train'],
        26: [(128, 128, 0),  'Tree'],
        27: [(192, 128, 192),'Truck/Bus'],
        28: [(64, 0, 64),    'Tunnel'],
        29: [(192, 192, 0),  'Vegetation Misc'],
        30: [(0, 0, 0),      'Void'],
        31: [(64, 192, 0),   'Wall']
          }
        
        pred_mask = seg_model.predict(image)
        pred_mask = pred_mask.numpy()
 
# The 1st label is the background class added by the model, but we can remove it for this dataset.
        pred_mask = pred_mask[:,:,:,1:]
   
# We also need to remove the batch dimension.
        pred_mask = np.squeeze(pred_mask)
        pred_mask_class = np.argmax(pred_mask, axis=-1)
      
        pred_mask_rgb = class_to_rgb(pred_mask_class, class_index) 
# Function to overlay a segmentation map on top of an RGB image.
        run_inference(image, seg_model)    
      avi_to_mp4("sem.avi","sem.mp4")
      st.video("sem.mp4")
      avi_to_mp4("sem_2.avi","sem_2.mp4")
      st.video("sem_2.mp4")
