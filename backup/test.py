import pygame
import cv2
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-Time Edge Detection Visualization")

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Pygame utilities
font = pygame.font.Font(None, 24)

# Function to create sliders
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, value, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.handle_rect = pygame.Rect(x, y, 10, h)

    def draw(self, screen):
        pygame.draw.rect(screen, (200, 200, 200), self.rect)
        handle_x = int(self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.w)
        self.handle_rect.x = handle_x - 5
        pygame.draw.rect(screen, (100, 100, 250), self.handle_rect)
        text = font.render(f"{self.label}: {self.value:.1f}", True, (255, 255, 255))
        screen.blit(text, (self.rect.x, self.rect.y - 25))

    def update(self, mouse_x):
        if self.rect.collidepoint(mouse_x, self.rect.y + self.rect.h // 2):
            relative_x = mouse_x - self.rect.x
            self.value = self.min_val + (relative_x / self.rect.w) * (self.max_val - self.min_val)
            self.value = max(self.min_val, min(self.value, self.max_val))
        return self.value

# Create sliders for both techniques
sliders = [
    Slider(50, 550, 400, 20, 0, 100, 50, "Threshold (Binary)"),
    Slider(50, 600, 400, 20, 50, 200, 100, "Canny Min Threshold"),
    Slider(50, 650, 400, 20, 100, 300, 150, "Canny Max Threshold")
]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get mouse input
    mouse_pressed = pygame.mouse.get_pressed()
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if mouse_pressed[0]:
        for slider in sliders:
            slider.update(mouse_x)

    # Extract slider values
    binary_thresh = int(sliders[0].value)
    canny_min = int(sliders[1].value)
    canny_max = int(sliders[2].value)

    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (WIDTH // 2, HEIGHT // 2))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Process frame with Thresholding
    _, thresh_frame = cv2.threshold(gray_frame, binary_thresh, 255, cv2.THRESH_BINARY)

    # Process frame with Canny Edge Detection
    canny_edges = cv2.Canny(gray_frame, canny_min, canny_max)

    # Draw everything in Pygame
    screen.fill((30, 30, 30))

    # Convert OpenCV images to Pygame surfaces
    original_surf = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
    thresh_surf = pygame.surfarray.make_surface(cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2RGB).swapaxes(0, 1))
    canny_surf = pygame.surfarray.make_surface(cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB).swapaxes(0, 1))

    # Display images
    screen.blit(original_surf, (0, 0))
    screen.blit(thresh_surf, (WIDTH // 2, 0))
    screen.blit(canny_surf, (WIDTH // 2, HEIGHT // 2))

    # Draw sliders
    for slider in sliders:
        slider.draw(screen)

    # Labels
    original_label = font.render("Original Frame", True, (255, 255, 255))
    thresh_label = font.render("Thresholding", True, (255, 255, 255))
    canny_label = font.render("Canny Edge Detection", True, (255, 255, 255))

    screen.blit(original_label, (20, 20))
    screen.blit(thresh_label, (WIDTH // 2 + 20, 20))
    screen.blit(canny_label, (WIDTH // 2 + 20, HEIGHT // 2 + 20))

    pygame.display.flip()

# Release the webcam and quit Pygame
cap.release()
pygame.quit()
