HoloControl Wired (Under Developement)

Major issue :- Too much latency because of transferring live feed 2 times over the network
Potential solution :- Turn everything offline using USB cables

=================
This project creates a hand-gesture controlled HUD for Windows using OpenCV, Mediapipe, and MSS.
It captures your desktop screen and overlays it on top of your webcam feed.
Using a finger pinch gesture (thumb + index), you can toggle the HUD (show/hide).

--------------------------------------------------
Features
--------------------------------------------------
- Webcam video feed with hand tracking
- Pinch gesture detection to toggle the HUD on/off
- Real-time desktop screen capture blended with webcam view
- Optimized screen capture (10 FPS) for performance
- Transparent overlay HUD with smooth blending

--------------------------------------------------
Requirements
--------------------------------------------------
Install dependencies with pip:

    pip install opencv-python mediapipe mss numpy

- Python 3.8 or higher recommended
- Works on Windows 10/11

--------------------------------------------------
How to Run
--------------------------------------------------
1. Save the script as holocontrol.py
2. Open a terminal in the folder where the script is saved
3. Run the script with:

    python holocontrol.py

4. A window named "HoloControl Wired" will appear.
   - Show your hand to the camera
   - Pinch your thumb and index finger together to toggle the HUD
   - Press the Q key to quit

--------------------------------------------------
Controls
--------------------------------------------------
- Pinch gesture → Toggle HUD (on/off)
- Q key → Quit the application

--------------------------------------------------
Demo (concept)
--------------------------------------------------
When HUD is active:
- The desktop screen (90% scale) is shown inside the webcam view
- It blends semi-transparently with your camera feed
- A black border HUD frame is drawn

When HUD is hidden:
- You only see the webcam feed with hand landmarks

--------------------------------------------------
Next Steps / Ideas
--------------------------------------------------
- Stream HUD feed to your phone via Flask server
- Add gesture controls to resize/move HUD
- Capture notifications or apps into HUD

