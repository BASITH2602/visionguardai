from flask import Flask, render_template, Response, request, send_file
import cv2
import random
from reportlab.pdfgen import canvas

app = Flask(__name__)

camera = cv2.VideoCapture(0)

# =========================
# LOAD HAAR CASCADE MODELS
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# =========================
# GLOBAL VARIABLES
# =========================

blink_count = 0
blink_frames = 0
eyes_currently_detected = True

frame_counter = 0
eye_strain_warning = False

distance_message = "Good Distance"

drowsy_warning = False

# =========================
# VISION TEST
# =========================

letters = ['E', 'F', 'P', 'T', 'O', 'Z', 'L', 'D']

current_letter = random.choice(letters)

vision_score = 0
total_attempts = 0

# =========================
# COLOR BLINDNESS TEST
# =========================

color_tests = [
    {"image": "plate1.png", "answer": "W"},
    {"image": "plate2.png", "answer": "28"},
    {"image": "plate3.png", "answer": "12"},
    {"image": "plate4.png", "answer": "6"},
    {"image": "plate5.png", "answer": "6"},
]

remaining_color_tests = color_tests.copy()

current_color_test = random.choice(remaining_color_tests)

remaining_color_tests.remove(current_color_test)

color_score = 0
color_attempts = 0

# =========================
# CAMERA PROCESSING
# =========================

def generate_frames():

    global blink_count
    global blink_frames
    global eyes_currently_detected
    global frame_counter
    global eye_strain_warning
    global distance_message
    global drowsy_warning

    while True:

        frame_counter += 1

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        eye_detected = False

        for (x, y, w, h) in faces:

            # Distance estimation
            face_area = w * h

            if face_area > 120000:
                distance_message = "WARNING: Too Close"

            elif face_area < 30000:
                distance_message = "WARNING: Move Closer"

            else:
                distance_message = "Good Distance"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            roi_gray = gray[y:y + h // 2, x:x + w]
            roi_color = frame[y:y + h // 2, x:x + w]

            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(30, 30)
            )

            if len(eyes) > 0:
                eye_detected = True

            for (ex, ey, ew, eh) in eyes:

                cv2.rectangle(
                    roi_color,
                    (ex, ey),
                    (ex + ew, ey + eh),
                    (0, 255, 0),
                    2
                )

        # Blink Detection
        if eye_detected:

            blink_frames = 0

            if not eyes_currently_detected:
                blink_count += 1

            eyes_currently_detected = True

        else:

            blink_frames += 1

            if blink_frames > 2:
                eyes_currently_detected = False

        # Drowsiness Detection
        if blink_frames > 15:
            drowsy_warning = True
        else:
            drowsy_warning = False

        # Eye Strain Detection
        if frame_counter > 300:

            if blink_count < 2:
                eye_strain_warning = True
            else:
                eye_strain_warning = False

            frame_counter = 0
            blink_count = 0

        # UI Text
        cv2.putText(
            frame,
            'VisionGuard AI',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f'Blinks: {blink_count}',
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        if eye_strain_warning:

            cv2.putText(
                frame,
                'WARNING: Eye Strain Detected',
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        cv2.putText(
            frame,
            distance_message,
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            3
        )

        if drowsy_warning:

            cv2.putText(
                frame,
                'DROWSINESS DETECTED',
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

# =========================
# HOME PAGE
# =========================

@app.route('/')
def index():

    accuracy = 0

    if total_attempts > 0:
        accuracy = int((vision_score / total_attempts) * 100)

    color_accuracy = 0

    if color_attempts > 0:
        color_accuracy = int((color_score / color_attempts) * 100)

    return render_template(
        'index.html',
        current_letter=current_letter,
        vision_score=vision_score,
        total_attempts=total_attempts,
        accuracy=accuracy,
        current_color_test=current_color_test,
        color_score=color_score,
        color_attempts=color_attempts,
        color_accuracy=color_accuracy
    )

# =========================
# VIDEO STREAM
# =========================

@app.route('/video')
def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================
# VISION TEST
# =========================

@app.route('/vision_test', methods=['POST'])
def vision_test():

    global current_letter
    global vision_score
    global total_attempts

    user_answer = request.form['answer'].upper()

    total_attempts += 1

    if user_answer == current_letter:
        vision_score += 1

    current_letter = random.choice(letters)

    return index()

# =========================
# COLOR TEST
# =========================

@app.route('/color_test', methods=['POST'])
def color_test():

    global current_color_test
    global color_score
    global color_attempts
    global remaining_color_tests

    user_answer = request.form['color_answer'].upper()

    color_attempts += 1

    if user_answer == current_color_test['answer']:
        color_score += 1

    if len(remaining_color_tests) == 0:
        remaining_color_tests = color_tests.copy()

    current_color_test = random.choice(remaining_color_tests)

    remaining_color_tests.remove(current_color_test)

    return index()

# =========================
# PDF REPORT GENERATION
# =========================

@app.route('/generate_report')
def generate_report():

    pdf_file = "VisionGuard_Report.pdf"

    c = canvas.Canvas(pdf_file)

    c.setFont("Helvetica-Bold", 22)
    c.drawString(150, 800, "VisionGuard AI Report")

    c.setFont("Helvetica", 14)

    c.drawString(50, 740, f"Vision Test Accuracy: {vision_score}/{total_attempts}")

    c.drawString(50, 700, f"Color Blindness Score: {color_score}/{color_attempts}")

    c.drawString(50, 660, f"Distance Status: {distance_message}")

    c.drawString(50, 620, f"Eye Strain Warning: {eye_strain_warning}")

    c.drawString(50, 580, f"Drowsiness Warning: {drowsy_warning}")

    c.drawString(50, 520, "AI Recommendation:")

    recommendation = "Maintain healthy screen distance and take regular eye breaks."

    c.drawString(70, 490, recommendation)

    c.save()

    return send_file(
        pdf_file,
        as_attachment=True
    )

# =========================
# MAIN
# =========================

if __name__ == '__main__':
    app.run(debug=True)