import os
import cv2
import torch
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
from playsound import playsound
from sort import Sort
import numpy as np

from graph_display import show_graph

# Initialize the main application window
root = tk.Tk()
root.title("Head Detection with YOLOv5 and SORT Tracking")
root.geometry("800x650")

# Make the window resizable only in height
root.resizable(width=False, height=True)

# Load the YOLOv5 model
def load_model():
    global model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        lbl_status.config(text="Model loaded successfully.")
        print("YOLOv5 model loaded successfully.")
    except Exception as e:
        lbl_status.config(text="Failed to load model.")
        print(f"Error loading YOLOv5 model: {e}")

threading.Thread(target=load_model).start()

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Global variables for video capture and stopping the thread
cap = None
stop_thread = False
alarm_playing = False
tracked_ids = set()  # Keep track of unique IDs ever encountered
current_ids = set()  # Keep track of IDs currently in the frame

def open_video():
    global cap, stop_thread
    stop_thread = False
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", ".mp4"), ("All files", ".*")))

    if video_path:
        print(f"Selected video path: {video_path}")  # Debugging output
        cap = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video.")
            print("Error: Could not open video.")  # Debugging output
        else:
            print("Video opened successfully.")  # Debugging output
            threading.Thread(target=detect_people).start()  # Start detection in a new thread
    else:
        print("No video file selected.")  # Debugging output

def start_webcam():
    global cap, stop_thread
    stop_thread = True  # Stop any previous detection thread
    time.sleep(0.1)  # Give time for previous threads to terminate
    if cap is not None:
        cap.release()  # Release previous video capture
    stop_thread = False  # Reset stop flag for new thread

    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        print("Error: Could not open webcam.")  # Debugging output
        return

    print("Webcam opened successfully.")  # Debugging output
    threading.Thread(target=detect_people, daemon=True).start()  # Start detection in a new thread

#Alarm will only sound once every 10 seconds
last_alarm_time = 0
alarm_interval = 10  

def play_alarm():
    global alarm_playing, last_alarm_time
    current_time = time.time()
    if not alarm_playing and (current_time - last_alarm_time > alarm_interval):
        alarm_playing = True
        last_alarm_time = current_time
        try:
            # Construct the path dynamically
            script_dir = os.path.dirname(os.path.abspath(_file_))
            alarm_path = os.path.join(script_dir, "alarm.mp3")
            playsound(alarm_path)
        except Exception as e:
            print(f"Error playing alarm: {e}")
        finally:
            alarm_playing = False


#starts to detect heads of people in every 5th frame of the vid
#remove try_except(if needed) to understand the rectangle box issue
def detect_people():
    try:
        global cap, stop_thread, tracked_ids, current_ids, roi_coords
        print("Starting people detection...")

        frame_count = 0
        process_interval = 5  # Process every 5 frames

        while cap.isOpened() and not stop_thread:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached or error reading the video!")
                break

            frame_count += 1
            if frame_count % process_interval != 0:
                continue

            try:
                frame = cv2.resize(frame, (640, 480))

                # Crop the frame to the selected ROI, if any
                if roi_coords:
                    x1, y1, x2, y2 = roi_coords
                    frame = frame[y1:y2, x1:x2]

                results = model(frame)

                if len(results.xyxy) == 0 or results.xyxy[0].shape[1] < 6:
                    print("Warning: Model output is invalid. Skipping frame.")
                    continue

                # Extract bounding box results for 'person' class
                people = results.xyxy[0].cpu().numpy()
                people = [p for p in people if int(p[5]) == 0]  # Class 0 corresponds to 'person'

                # Prepare detections for SORT tracker
                detections = []
                for x1, y1, x2, y2, conf, _ in people:
                    # Adjust coordinates for cropped ROI
                    if roi_coords:
                        x1 += roi_coords[0]
                        x2 += roi_coords[0]
                        y1 += roi_coords[1]
                        y2 += roi_coords[1]
                    detections.append([x1, y1, x2, y2, conf])

                # Update the SORT tracker
                tracked_objects = tracker.update(np.array(detections))

                current_ids = set()
                for x1, y1, x2, y2, track_id in tracked_objects:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    track_id = int(track_id)
                    current_ids.add(track_id)

                    # Adjust bounding boxes back for cropped frames
                    if roi_coords:
                        x1 -= roi_coords[0]
                        x2 -= roi_coords[0]
                        y1 -= roi_coords[1]
                        y2 -= roi_coords[1]

                    # Draw a bounding box for the head region
                    head_height = int(0.25 * (y2 - y1))
                    head_y2 = y1 + head_height
                    cv2.rectangle(frame, (x1, y1), (x2, head_y2), (0, 165, 255), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


                    # Track the current frame IDs
                    current_ids.add(track_id)
                    tracked_ids.add(track_id)

                # Update tracked IDs # above two lines handles the updating of the current_ids and tracked_ids
                # tracked_ids.update(current_ids)

                # Update GUI labels
                lbl_total_count.config(text=f"Total People Appeared: {len(tracked_ids)}")
                lbl_current_count.config(text=f"Current People in Frame: {len(current_ids)}")

                # Play the alarm if more than 10 unique people are detected
                if len(tracked_ids) > 250:
                    threading.Thread(target=play_alarm).start()

                # Display the frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                lbl_video.imgtk = imgtk
                lbl_video.configure(image=imgtk)

                root.update()
                time.sleep(0.03)
            except Exception as inner_e:
                print(f"Error processing frame: {inner_e}")

        cap.release()
        cv2.destroyAllWindows()
        print("Video capture released.")

    except Exception as e:
        print(f"Error in detection loop: {e}")



# Global dictionary to store minute counts
minute_counts = {}
tracking_active = True

def track_minute_counts():
    #Track the minute counts in a background thread.
    global minute_counts, tracked_ids, tracking_active
    while tracking_active:
        # Initialize the current minute tracking
        current_minute = time.strftime("%Y-%m-%d %H:%M")
        current_minute_ids = set()

        # Sleep for a second to synchronize with the actual minute
        time.sleep(1)

        while time.strftime("%Y-%m-%d %H:%M") == current_minute and tracking_active:
            # Add IDs currently in frame to the minute's ID set
            current_minute_ids.update(current_ids)
            time.sleep(0.5)  # Small delay to avoid too much CPU usage

        # After the minute is over, save the count and reset
        minute_counts[current_minute] = len(current_minute_ids)
        print(f"{current_minute}: {minute_counts[current_minute]} people detected.")  # Debug output


def display_minute_counts():

    #Display the stored minute counts as a text summary and graph in a new window.

    if not minute_counts:
        messagebox.showinfo("Minute Counts", "No data recorded yet.")
        return

    # Prompt the user to select the graph type
    response = messagebox.askquestion("Graph Type", "Would you like a line graph? (Select 'No' for bar graph)")
    graph_type = "line" if response == "yes" else "bar"

    # Display the graph and the summary
    show_graph(minute_counts, graph_type=graph_type)


# Start the minute tracking in a separate thread
minute_thread = threading.Thread(target=track_minute_counts, daemon=True)
minute_thread.start()




roi_coords = None  # Global variable to store ROI coordinates

def select_roi():
    global cap, roi_coords
    if cap is None or not cap.isOpened():
        messagebox.showerror("Error", "Video or webcam is not active.")
        return

    # Read a single frame to select ROI
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Unable to capture frame for ROI selection.")
        return

    # Resize the frame for consistent display
    frame = cv2.resize(frame, (640, 480))

    # OpenCV ROI selection
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    if roi == (0, 0, 0, 0):
        roi_coords = None
        messagebox.showinfo("Info", "ROI selection cleared.")
    else:
        x, y, w, h = roi
        roi_coords = (x, y, x + w, y + h)
        messagebox.showinfo("Info", f"ROI selected: {roi_coords}")




# Function for clearing the ROI setting or choice
def set_roi(coords):
    global roi_coords
    roi_coords = coords
    messagebox.showinfo("Info", "ROI cleared and reset to full frame.")




def stop_detection():
    global stop_thread, tracked_ids, current_ids
    stop_thread = True

    tracking_active = False  # Stop the minute count tracking

    lbl_video.configure(image='')  # Clear the video display
    lbl_total_count.config(text="Total People Appeared: 0")  # Reset total count display
    lbl_current_count.config(text="Current People in Frame: 0")  # Reset current count display
    tracked_ids.clear()  # Clear tracked IDs
    current_ids.clear()  # Clear current IDs
    print("Detection stopped.")  # Debugging output

# Creating GUI elements
lbl_video = tk.Label(root)
lbl_video.pack()

lbl_total_count = tk.Label(root, text="Total People Appeared: 0", font=("Arial", 14))
lbl_total_count.pack()

lbl_current_count = tk.Label(root, text="Current People in Frame: 0", font=("Arial", 14))
lbl_current_count.pack()

lbl_status = tk.Label(root, text="Loading model...", font=("Arial", 14))
lbl_status.pack()

# Frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

btn_open_video = tk.Button(button_frame, text="Open Video", command=open_video)
btn_open_video.pack(side=tk.LEFT, padx=10)

btn_webcam = tk.Button(button_frame, text="Start Webcam", command=start_webcam)
btn_webcam.pack(side=tk.LEFT, padx=10)

btn_stop = tk.Button(button_frame, text="Stop Detection", command=stop_detection)
btn_stop.pack(side=tk.LEFT, padx=10)


# Add a display button to the GUI
btn_display_counts = tk.Button(button_frame, text="Display Minute Counts", command=display_minute_counts)
btn_display_counts.pack(side=tk.LEFT, padx=10)

# ROI (Region of Interest) Selection GUI
btn_select_roi = tk.Button(button_frame, text="Select ROI", command=select_roi)
btn_select_roi.pack(side=tk.LEFT, padx=10)

# Clear the ROI
btn_clear_roi = tk.Button(button_frame, text="Clear ROI", command=lambda: set_roi(None))
btn_clear_roi.pack(side=tk.LEFT, padx=10)



# Run the main Tkinter loop
root.mainloop()
# ROI(Region of Interest), solved the ROI bugs and made the ROI selection count more accurate
