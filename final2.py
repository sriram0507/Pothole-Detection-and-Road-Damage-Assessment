import cv2
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.solutions import distance_calculation
from ultralytics.solutions import object_counter
from collections import deque
import numpy as np
import tempfile


# Function to resize frames
def resize_frame(frame, target_width, target_height):
    return cv2.resize(frame, (target_width, target_height))


def get_object_size(class_index):  # Assuming class_index corresponds to object type
    # Replace this with your logic to return the actual size of the object based on class_index
    # This example assumes all objects are 1 meter tall
    return 1.0


def calculate_relative_distance(center_y, image_height, object_size):
    # Assuming objects are roughly upright and facing the camera
    # This calculates a proxy for relative distance based on object size and position in the frame
    distance_proxy = (image_height - center_y) * image_height / object_size
    return distance_proxy / 1000


# Function to process the uploaded video
def process_video(
    video_file,output_video_path, model, names, dist_obj, line_points, counter, damage_deque,target_width, target_height
):

    # Define font and text position for displaying road damage percentage
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color for text
    text_position = (40, 80)
    text_position_potholes = (40, 120)  # Position for displaying total potholes

    cap = cv2.VideoCapture(video_file)
    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )
    
    # Resize frame if dimensions don't match target
    if w != target_width or h != target_height:
        target_size = (target_width, target_height)
    else:
        target_size = (w, h)

    # Video writer for object tracking
    out_object_tracking = cv2.VideoWriter(
        "instance-segmentation-object-tracking.mp4",
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w, h),
    )

    out_road_damage = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"avc1"), fps, target_size
    )

    while True:
        ret, im0 = cap.read()
        if not ret:
            print(
                "Video frame is empty or video processing has been successfully completed."
            )
            break
        
        # Resize frame if dimensions don't match target
        if w != target_width or h != target_height:
            im0 = resize_frame(im0, target_width, target_height)

        image_height, _ = im0.shape[:2]
        # Run object detection
        detections = model(im0)

        # Perform instance segmentation object tracking
        results = model.track(im0, persist=True)

        # Process object tracking results
        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotator = Annotator(im0, line_width=2)

            for mask, track_id in zip(masks, track_ids):
                annotator.seg_bbox(
                    mask=mask,
                    mask_color=colors(track_id, True),
                    track_label=str(track_id),
                )

            im0 = annotator.result()

            out_object_tracking.write(im0)

            # Update the final track ID value
            final_track_id = len(track_ids)
        else:
            final_track_id = None

        # Process object detection results
        if detections:
            detection_result = detections[0]
            im0 = detection_result.plot()

        # Calculate road damage percentage
        percentage_damage = 0
        if results[0].masks is not None:
            total_area = 0
            masks = results[0].masks.data.cpu().numpy()
            image_area = (
                im0.shape[0] * im0.shape[1]
            )  # Total number of pixels in the image
            for mask in masks:
                binary_mask = (mask > 0).astype(np.uint8) * 255
                contour, _ = cv2.findContours(
                    binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                total_area += cv2.contourArea(contour[0])

            percentage_damage = (total_area / image_area) * 1000

        # Calculate and update the percentage damage
        damage_deque.append(percentage_damage)
        smoothed_percentage_damage = sum(damage_deque) / len(damage_deque)

        # Draw a thick line for text background

        if detections:
            detection_result = detections[0]
            im0 = detection_result.plot()

            # Iterate through detections (bounding boxes)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[
                        0
                    ]  # Assuming xyxy format for bounding box coordinates
                    x1, y1, x2, y2 = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    )  # Convert to integers for OpenCV

                    w, h = (
                        x2 - x1,
                        y2 - y1,
                    )  # Calculate width and height of bounding box

                    class_index = int(
                        box.cls[0]
                    )  # Assuming class index is stored in box.cls[0]

                    # Calculate center point of the bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    object_size = get_object_size(class_index)

                    # Assuming calculate_distance_to_line function exists
                    distance_proxy = calculate_relative_distance(
                        center_y, image_height, object_size
                    )
                    # Display distance on the frame (optional)
                    cv2.rectangle(
                        im0, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )  # Draw bounding box
                    cv2.circle(
                        im0, (center_x, center_y), 5, (0, 0, 255), -1
                    )  # Draw center point
                    cv2.putText(
                        im0,
                        f"Distance appx: {distance_proxy:.2f} cm",
                        (x1 + 70, y1 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.line(
                        im0,
                        (text_position[0], text_position[1] - 10),
                        (text_position[0] + 350, text_position[1] - 10),
                        (0, 0, 255),
                        40,
                    )

                    # Annotate the frame with the percentage of damage
                    cv2.putText(
                        im0,
                        f"Road Damage: {percentage_damage:.2f}%",
                        text_position,
                        font,
                        font_scale,
                        font_color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        im0,
                        f"Total Potholes: {final_track_id}",
                        text_position_potholes,
                        font,
                        font_scale,
                        font_color,
                        2,
                        cv2.LINE_AA,
                    )

        # Write the frame to the output video for road damage assessment
        out_road_damage.write(im0)

        # Display the processed frame on the screen
        cv2.imshow("Combined Processing", im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video capture and video write objects
    cap.release()
    out_object_tracking.release()
    out_road_damage.release()
    
    return track_ids,smoothed_percentage_damage


# Main function to run the Streamlit app
def main():
    # Load YOLO model
    model = YOLO("best (1).pt")
    names = model.model.names

    # Video writer for object tracking
    # out_object_tracking = cv2.VideoWriter('instance-segmentation-object-tracking.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Initialize distance calculation object
    dist_obj = distance_calculation.DistanceCalculation()
    dist_obj.set_args(names=names, view_img=True)

    # Define line points for object counting
    line_points = [(0, 600), (1400, 600)]
    counter = object_counter.ObjectCounter()
    counter.set_args(
        view_img=True,
        reg_pts=line_points,
        classes_names=names,
        draw_tracks=True,
        line_thickness=2,
    )

    # Video writer for road damage assessment
    # out_road_damage = cv2.VideoWriter('road_damage_assessment.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

    # Initialize a deque with fixed length for averaging the last 10 percentage damages
    damage_deque = deque(maxlen=100)

    # Set title of the Streamlit app
    st.title("Pothole detection and road damage assessment")

    # File uploader for first video
    video_file1 = st.file_uploader("Upload First Video File", type=["mp4"])
    if video_file1 is not None:
        st.video(video_file1)

    # File uploader for second video
    video_file2 = st.file_uploader("Upload Second Video File", type=["mp4"])
    if video_file2 is not None:
        st.video(video_file2)

    if st.button("Process Videos"):
        if video_file1 is not None and video_file2 is not None:
            # Create temporary files for processing
            with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file1, tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file2:
                temp_file1.write(video_file1.read())
                temp_file2.write(video_file2.read())
                temp_file1.seek(0)
                temp_file2.seek(0)

                # Get dimensions of first video
                cap = cv2.VideoCapture(temp_file1.name)
                target_width, target_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            
                # Process uploaded video
                track_ids1,smoothed_percentage_damage1=process_video(
                    temp_file1.name, "output1.mp4",model, names, dist_obj, line_points, counter, damage_deque , target_width, target_height
                )
                
                track_ids2,smoothed_percentage_damage2=process_video(
                    temp_file2.name, "output2.mp4",model, names, dist_obj, line_points, counter, damage_deque , target_width, target_height
                )

            st.success("Videos processed successfully!")
        
            # Display processed videos
            st.subheader("Processed Video 1")
            video_file1_output = open("output1.mp4", "rb").read()
            st.video(video_file1_output)
            cnt1=max(track_ids1)
            st.write("TOTAL NO.OF POTHOLES:",cnt1)
            st.write('TOTAL DAMAGE IN THE ROAD : ', smoothed_percentage_damage1)
            
            st.subheader("Processed Video 2")
            video_file2_output = open("output2.mp4", "rb").read()
            st.video(video_file2_output)     
            cnt2=abs(max(track_ids2)-max(track_ids1))
            st.write("TOTAL NO.OF POTHOLES:",cnt2)
            st.write('TOTAL DAMAGE IN THE ROAD : ', smoothed_percentage_damage2)
            if(cnt1>cnt2):
                st.write('Best route to take based on potholes is route 2')
            else:
                st.write('Best route to take based on potholes is route 1')
            if(smoothed_percentage_damage1>smoothed_percentage_damage2): 
                st.write('Best route to take based on damage is route 2')
            else:
                st.write('Best route to take based on potholes is route 1')
        else:
            st.error("Please upload both video files.")
        

# Entry point of the Streamlit app
if __name__ == "__main__":
    main()
