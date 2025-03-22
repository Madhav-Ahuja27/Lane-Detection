
import cv2
import numpy as np

def grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image, kernel_size=5):
    """Apply Gaussian Blur to smooth the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_edge(image, low_threshold=50, high_threshold=150):
    """Apply Canny Edge Detection."""
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image):
    """Mask the region of interest to focus on lane lines."""
    height, width = image.shape[:2]
    mask = np.zeros_like(image)

    # Define a trapezoidal region to focus on road lanes
    region = np.array([[
        (width * 0.1, height), 
        (width * 0.45, height * 0.6), 
        (width * 0.55, height * 0.6), 
        (width * 0.9, height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, region, 255)
    return cv2.bitwise_and(image, mask)

def hough_lines(image):
    """Detect lane lines using Hough Transform."""
    return cv2.HoughLinesP(image, 1, np.pi/180, 50, minLineLength=100, maxLineGap=160)

def draw_lines(image, lines):
    """Draw lane lines on the image."""
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return cv2.addWeighted(image, 0.8, line_image, 1, 0)

def process_frame(frame):
    """Process a single video frame and detect lane lines."""
    gray = grayscale(frame)
    blurred = gaussian_blur(gray)
    edges = canny_edge(blurred)
    roi = region_of_interest(edges)
    lines = hough_lines(roi)
    return draw_lines(frame, lines)

def process_video(input_path, output_path):
    """Process a video and detect lane lines."""
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        out.write(processed_frame)
        cv2.imshow("Lane Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = "project_video.mp4"  # Change this to your actual video file
    output_video = "output_video.avi"

    process_video(input_video, output_video)


#Convert to Grayscale
# Apply Gaussian Blur
# Edge Detection using Canny Algorithm
# Region of Interest (ROI) Masking
# Hough Transform for Line Detection
# Draw Lane Lines
# Process Video Frame-by-Frame
# Save and Display Processed Video
