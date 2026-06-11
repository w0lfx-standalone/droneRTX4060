import streamlit as st
import cv2
import tempfile
import time
import os
import pygame
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# ---confg---
# model path
MODEL_PATH = 'runs/detect/runs/detect/droneRTX/weights/best.pt' 
ALERT_SOUND = 'assets/alert.mp3'
HUNT_FOLDER = 'hunts'

# folder config
os.makedirs(HUNT_FOLDER, exist_ok=True)
os.makedirs('assets', exist_ok=True)

# Setup Page Layout
st.set_page_config(
    page_title="DroneRTX",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sound sys
if "sound_init" not in st.session_state:
    pygame.mixer.init()
    st.session_state["sound_init"] = True

def play_alert():
  
    if os.path.exists(ALERT_SOUND):
        if not pygame.mixer.music.get_busy():
            try:
                pygame.mixer.music.load(ALERT_SOUND)
                pygame.mixer.music.play(loops=-1)
            except Exception as e:
                st.toast(f"Audio Error: {e}")

def stop_alert():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# Load model
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Model not found at {MODEL_PATH}. Please check the path.")
    st.stop()

# ---side ctrl--
with st.sidebar:
#    st.image("https://img.icons8.com/fluency/96/drone.png", width=80)
    st.title("DroneRTX\nControlRoom")
    st.markdown("---")
    
    # - nput Source
    input_source = st.radio("Source", ["Live Feed", "Upload Video/Photo"])
    
    # - live camera settings
    source_index = 0
    if input_source == "Live Feed":
        cam_type = st.selectbox("Device Type", ["Laptop Webcam", "Phone (IP Camera)"])
        
        if cam_type == "Phone (IP Camera)":
            cam_url = st.text_input("IP Camera URL", "http://192.168.1.XX:8080/video")
            source_index = cam_url
        else:
            source_index = 0 # Laptop Webcam
    
    st.markdown("---")
    
    # - settings
    conf_thresh = st.slider("Sensitivity Threshold", 0.1, 1.0, 0.50)
    enable_audio = st.toggle("Alarm", value=True)
    enable_save = st.toggle("Auto-Save Detection", value=True)
    
    st.markdown("---")
    
    # - stats
    kpi1, kpi2 = st.columns(2)
    st_fps = kpi1.metric("FPS", "0")
    st_count = kpi2.metric("Threats", "0")
    
    if st.button("Refresh Evidence Gallery"):
        st.rerun()

# ---main page---
st.markdown("### 🚨 Real-Time Surveillance")

# ui placeholder
alert_banner = st.empty()
video_placeholder = st.empty()

# ---big brain---
if input_source == "Live Feed":
    run_btn = st.checkbox("🔳ACTIVATE SYSTEM🔳", value=True)
    
    if run_btn:
        cap = cv2.VideoCapture(source_index)
        
        if not cap.isOpened():
            st.error("❌ Error: Could not open video source. Check IP or webcam index.")
            st.stop()
            
        prev_time = 0
        save_cooldown = 0
        
        while run_btn:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Video Signal Lost.")
                break
                
            frame = cv2.resize(frame, (1280, 720))
            
            results = model.predict(frame, conf=conf_thresh, verbose=False)
            
            # --- detection brain ---
            drone_detected = False
            drone_count = 0
            
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0: 
                    drone_detected = True
                    drone_count += 1
            
            # --- alerts ---
            if drone_detected:
                # -- visual alert
                alert_banner.markdown(
                    """<div style='background-color: #ff4b4b; padding: 10px; border-radius: 5px; text-align: center;'>
                    <h2 style='color: white; margin:0;'>🚨 DRONE DETECTED 🚨</h2></div>""", 
                    unsafe_allow_html=True
                )
                
                # -- audio alert
                if enable_audio:
                    play_alert()
                
                # -- evidence autosave 
                if enable_save and (time.time() - save_cooldown > 2.0):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{HUNT_FOLDER}/hunt_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    st.toast(f"Evidence Captured: {filename}")
                    save_cooldown = time.time()
            
            else:
                stop_alert()
                alert_banner.info("✅ Airspace Clear - Scanning...")

            # --- 4. RENDER OUTPUT ---
            annotated_frame = results[0].plot()
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # FPS calc
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # updates
            st_fps.metric("FPS", f"{int(fps)}")
            st_count.metric("Threats", f"{drone_count}")

elif input_source == "Upload Video/Photo":
    uploaded_file = st.file_uploader("Upload Footage", type=['mp4', 'avi', 'mov', 'jpg', 'png'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # image checking
        if uploaded_file.type in ['image/jpeg', 'image/png']:
            img = cv2.imread(tfile.name)
            results = model.predict(img, conf=conf_thresh)
            res_plotted = results[0].plot()
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Analysis Result", use_container_width=True)
        
        # video checking
        else:
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                results = model.predict(frame, conf=conf_thresh)
                res_plotted = results[0].plot()
                st_frame.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), use_container_width=True)

# --- evidence gallery ---
st.markdown("---")
st.subheader("🗂️ Evidence Locker")

# image sorter
images = sorted([f for f in os.listdir(HUNT_FOLDER) if f.endswith(('.jpg', '.png'))], reverse=True)

if images:
    # 8 at a time
    cols = st.columns(4)
    for idx, img_name in enumerate(images[:8]):
        img_path = os.path.join(HUNT_FOLDER, img_name)
        with cols[idx % 4]:
            st.image(img_path, caption=img_name.split('.')[0], use_container_width=True)
else:
    st.caption("No detections recorded yet.")