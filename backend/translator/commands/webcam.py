# backend/translator/webcam.py
import absl.logging
import cv2
import typer

from translator.landmarks import draw_landmarks_on_frame
from translator.tf import PJMClassifier

app = typer.Typer()

# Disable MediaPipe logs (INFO and WARNING)
absl.logging.set_verbosity(absl.logging.ERROR)


@app.command()
def start() -> None:
    """
    Test hand landmark detection and PJM gesture classification using a webcam feed.
    Displays the video feed with detected hand landmarks and predicted letter.
    Prints detected hand landmarks and predictions in the terminal.
    Press 'q' to exit.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize classifier with error handling
    classifier = PJMClassifier()
    try:
        classifier.load_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Cannot proceed without the trained model or hand landmarker model.")
        return
    except Exception as e:
        print(f"Error loading classifier model: {e}")
        return

    print("Starting webcam feed. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Extract landmarks and classify with error handling
        try:
            predicted_letter, detection_result = classifier.process_frame(frame)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Cannot continue inference without the hand landmarker model.")
            break
        except Exception as e:
            print(f"Error processing frame: {e}")
            predicted_letter, detection_result = None, None  # Continue to the next frame

        # Draw landmarks on the frame
        if detection_result and detection_result.hand_landmarks:
            # Convert frame to RGB for drawing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Draw landmarks and connections using MediaPipe's drawing utility
            frame_annotated = draw_landmarks_on_frame(frame_rgb, detection_result)
            frame = frame_annotated

            # Print landmark details
            print(f"Detected {len(detection_result.hand_landmarks)} hand(s):")
            for i, hand in enumerate(detection_result.hand_landmarks):
                print(f"Hand {i + 1}: {len(hand)} landmarks")
                for j, point in enumerate(hand):
                    print(f"  Landmark {j}: (x: {point.x:.2f}, y: {point.y:.2f}, z: {point.z:.2f})")
            if predicted_letter:
                print(f"Predicted PJM letter: {predicted_letter}")
        else:
            print("No hands detected.")

        # Display the predicted letter in the top-right corner
        height, width, _ = frame.shape
        text = f"Letter: {predicted_letter if predicted_letter else 'None'}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 255, 0)  # Green
        # Calculate text size to position it in the top-right corner
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = width - text_width - 10  # 10 pixels padding from the right edge
        text_y = 10 + text_height  # 10 pixels padding from the top edge
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        # Display the frame with landmarks and prediction
        cv2.imshow("Webcam Feed", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
