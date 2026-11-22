import time
import threading
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
import pyaudio
import pyvirtualcam
from utils import prepare_sprite

# === You can safely change and tune the values below this comment ===

# Dimensions of the virtual camera
CAM_WIDTH, CAM_HEIGHT = 640, 480

# The frame rate of the virtual camera
# Corresponds to how quickly the sprite will update
# Higher numbers = more CPU usage
FPS = 30

# The volume threshold used to determine when the character is speaking
# This must be tuned as needed, as different setups will have different results
SILENCE_THRESHOLD = 50

# Determines how much your eyes must be closed to register as a blink
# Set between 0 and 1 (tune as needed)
BLINKING_THRESHOLD = 0.5

# Determines how wide your mouth must be open to register as scared
# Set between 0 and 1 (tune as needed)
SCARED_THRESHOLD = 0.5

# The sprites to be used for each expression
# The keys are (scared, speaking, blinking)
# Only update the paths to the sprites in the "images" folder to the sprites you are using
# The defualt expression key is (False, False, False): not scared, not speaking, not blinking
# The defualt expression must be set. Any unused expressions can be removed and will use the defualt instead
sprites = {
    (False, False, False): prepare_sprite(r"images\happy-normal-open.png", CAM_WIDTH, CAM_HEIGHT),
    (False, False, True):  prepare_sprite(r"images\happy-normal-closed.png", CAM_WIDTH, CAM_HEIGHT),
    (False, True,  True):  prepare_sprite(r"images\happy-speaking-closed.png", CAM_WIDTH, CAM_HEIGHT),
    (False, True,  False): prepare_sprite(r"images\happy-speaking-open.png", CAM_WIDTH, CAM_HEIGHT),
    (True,  True,  True):  prepare_sprite(r"images\scared-speaking-closed.png", CAM_WIDTH, CAM_HEIGHT),
    (True,  True,  False): prepare_sprite(r"images\scared-speaking-open.png", CAM_WIDTH, CAM_HEIGHT),
    (True,  False, True):  prepare_sprite(r"images\scared-normal-closed.png", CAM_WIDTH, CAM_HEIGHT),
    (True,  False, False): prepare_sprite(r"images\scared-normal-open.png", CAM_WIDTH, CAM_HEIGHT),
}

SHOW_RAW_VIDEO_FEED = False # Set to True if you want to see the raw video feed
SHOW_LANDMARKED_VIDEO_FEED = False # Set to True if you want to see the video feed with landmarks
# Note that displaying the video feeds may reduce performace

# === You shouldn't change anything below this comment unless you know what you're doing ===

assert sprites[(False, False, False)] is not None, "The default sprite (False, False, False) must be set"

# Audio setup
CHUNK = 1024
RATE = 44100
latest_rms = 0.0
audio_lock = threading.Lock()

p = pyaudio.PyAudio()
print(f"Using audio input device: \"{p.get_default_input_device_info()['name']}\"")

stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

stop_audio = threading.Event()
def audio_thread(): # continously updates the last recorded volume from the input deivce
    global latest_rms
    while not stop_audio.is_set():
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(data**2))
        with audio_lock:
            latest_rms = rms

threading.Thread(target=audio_thread, daemon=True).start() # Runs until the main thread stops

def is_speaking() -> bool:
    # Returns true if the volume is above the threshold
    with audio_lock:
        # print(latest_rms)
        return latest_rms > SILENCE_THRESHOLD

cap = cv2.VideoCapture(0) # start the video capture

running = True

current_sprite = sprites[(False, False, False)]
sprite_lock = threading.Lock()

FONT = cv2.FONT_HERSHEY_SIMPLEX

def on_result(result, output_image: mp.Image, timestamp_ms: int):
    """ Callback function that runs every time a new frame is processed """
    global running

    if output_image is None:
        return

    # print(f"Pose landmarks at {timestamp_ms} ms: {len(result.pose_landmarks)}")

    rgb_image = output_image.numpy_view()
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if SHOW_RAW_VIDEO_FEED: cv2.imshow('Video Stream', bgr_image) # Display the unmodified video stream

    if SHOW_LANDMARKED_VIDEO_FEED and result.pose_landmarks:
        mp_drawing.draw_landmarks(bgr_image, result.pose_landmarks, PoseLandmarksConnections.POSE_LANDMARKS)

    if result.face_blendshapes:
        # Get the current expression states
        eyeBlinkLeft = next((b.score for b in result.face_blendshapes if b.category_name == "eyeBlinkLeft"), 0)
        eyeBlinkRight = next((b.score for b in result.face_blendshapes if b.category_name == "eyeBlinkRight"), 0)
        blinking = eyeBlinkLeft > 0.5 or eyeBlinkRight > 0.5
        scared = next((b.score for b in result.face_blendshapes if b.category_name == "jawOpen"), 0) > 0.5
        speaking = is_speaking()

        if SHOW_LANDMARKED_VIDEO_FEED:
            # Add labels of the current detected states to the video
            cv2.putText(bgr_image, f"Blinking: {blinking}", (10, 30), FONT, 1, (0, 255, 0), 2)
            cv2.putText(bgr_image, f"Speaking: {speaking}", (10, 70), FONT, 1, (0, 255, 0), 2)
            cv2.putText(bgr_image, f"Scared: {scared}", (10, 110), FONT, 1, (0, 255, 0), 2)

        sprite = sprites.get((scared, speaking, blinking), sprites[(False, False, False)]) # Get sprite corresponding to the state
        with sprite_lock:
            # Update the sprite used by the virtual cam
            global current_sprite
            current_sprite = sprite

    if SHOW_LANDMARKED_VIDEO_FEED: cv2.imshow('Landmarks', bgr_image) # Update the video stream with landmarks

    if cv2.waitKey(10) & 0xFF == ord('q'):
        running = False

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="holistic_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
    output_face_blendshapes=True,
    result_callback=on_result,
)

def cleanup():
    """ Stops the video capture and closes all windows """
    print("Exiting...")
    stop_audio.set()
    cap.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    p.terminate()

# Main loop which captures the video and updates the virtual camera
with pyvirtualcam.Camera(width=CAM_WIDTH, height=CAM_HEIGHT, fps=30) as cam:
    with HolisticLandmarker.create_from_options(options) as landmarker:
        print("Use \"Ctr+C\" to close.")
        sprite = sprites[(False, False, False)]
        while running:
            try:
                # Async: automatically drops incoming camera frames when busy so results stay current
                ret, frame = cap.read()
                if not ret: break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(time.time() * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                landmarker.detect_async(mp_image, timestamp_ms)

                # Send the current sprite to the virtual camera
                with sprite_lock:
                    if current_sprite is not sprite:
                        sprite = current_sprite
                cam.send(sprite)
                cam.sleep_until_next_frame()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)
                break

# Cleanup when program exits
cleanup()