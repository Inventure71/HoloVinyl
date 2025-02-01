import os
import json
import cv2
import pygame
from utils.pygame_utils.Button import Button  # Adjust the import as needed

# --- Callback definitions ---
def button_clicked(n):
    print("Button", n, "clicked")

# Map callback identifier strings to functions.
CALLBACK_MAPPING = {
    "button_clicked": button_clicked,
}

# --- DigitalButtonEditor class ---
class DigitalButtonEditor:
    # Constants for resizing buttons
    RESIZE_HANDLE_SIZE = 10
    MIN_WIDTH = 30
    MIN_HEIGHT = 30

    def __init__(self, background_frame, save_file="variables/digital_buttons.json", font=None,
                 max_display_dimension=1024):
        """
        :param background_frame: A camera frame, either as a cv2 image (numpy array) or a pygame.Surface.
        :param save_file: File where digital button configurations are saved.
        :param font: Pygame font to use. If None, a default font is created.
        :param max_display_dimension: The size (in pixels) to which the longest side of the image will be scaled.
        """
        self.save_file = save_file
        self.font = font or pygame.font.Font(None, 32)
        self.buttons = []
        self.running = True

        # For moving buttons:
        self.dragging_button = None
        self.drag_offset = (0, 0)

        # For resizing buttons:
        self.resizing_button = None

        # For assigning unique callback parameters to new buttons.
        self.button_counter = 0

        # Instruction overlay.
        self.instruction_lines = [
            "Left-click on empty area: Add button",
            "Left-click on a button (outside resize handle): Drag to move",
            "Left-click on a button's lower-right square: Resize button",
            "Right-click on a button: Remove button",
            "Press 'S' to Save & Exit",
            "Press 'ESC' to Exit without saving"
        ]

        # Process the input frame.
        # If it's already a pygame.Surface, use it directly; otherwise, assume it's a cv2 image.
        if isinstance(background_frame, pygame.Surface):
            self.original_surface = background_frame.copy()
            original_width, original_height = self.original_surface.get_size()
        else:
            # Assume cv2 frame (BGR); convert to RGB.
            background_rgb = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
            self.original_surface = pygame.image.frombuffer(
                background_rgb.tobytes(),
                (background_rgb.shape[1], background_rgb.shape[0]),
                "RGB"
            )
            original_width, original_height = self.original_surface.get_size()

        self.original_size = (original_width, original_height)
        # Determine the scaling factor using only one maximum dimension.
        longest_side = max(original_width, original_height)
        self.scale_factor = max_display_dimension / longest_side

        display_width = int(original_width * self.scale_factor)
        display_height = int(original_height * self.scale_factor)
        self.background = pygame.transform.scale(self.original_surface, (display_width, display_height))

        # Load any previously saved buttons.
        self.load_buttons()

        """Run the digital button editor loop."""
        self.screen = pygame.display.set_mode(self.background.get_size())
        self.clock = pygame.time.Clock()

    def load_buttons(self):
        """
        Load digital buttons from the save file (if it exists) and scale them to display coordinates.
        Each saved button should include callback_id and callback_param.
        """
        if os.path.exists(self.save_file):
            with open(self.save_file, "r") as f:
                data = json.load(f)
            self.buttons = []
            for item in data:
                # Convert saved (original) coordinates to display coordinates.
                display_x = int(item.get("x", 0) * self.scale_factor)
                display_y = int(item.get("y", 0) * self.scale_factor)
                display_width = int(item.get("width", 100) * self.scale_factor)
                display_height = int(item.get("height", 50) * self.scale_factor)
                callback_id = item.get("callback_id", "button_clicked")
                callback_param = item.get("callback_param", 0)
                btn = Button(
                    x=display_x,
                    y=display_y,
                    width=display_width,
                    height=display_height,
                    text=item.get("text", "Digital Button"),
                    font=self.font,
                    text_color=tuple(item.get("text_color", [255, 255, 255])),
                    button_color=tuple(item.get("button_color", [0, 128, 255])),
                    hover_color=tuple(item.get("hover_color", [0, 102, 204])),
                    callback=lambda param=callback_param, cid=callback_id: CALLBACK_MAPPING[cid](param),
                    border_radius=item.get("border_radius", 15)
                )
                # Store callback info for later saving.
                btn.callback_id = callback_id
                btn.callback_param = callback_param
                self.buttons.append(btn)
                # Keep the counter in sync (if a loaded button has a higher callback_param).
                if callback_param >= self.button_counter:
                    self.button_counter = callback_param + 1
        else:
            self.buttons = []

    def save_buttons(self):
        """
        Save the current digital buttons to a JSON file.
        Button positions and sizes are converted back to the original scale.
        Also save the callback information.
        """
        data = []
        for btn in self.buttons:
            original_x = int(btn.rect.x / self.scale_factor)
            original_y = int(btn.rect.y / self.scale_factor)
            original_width = int(btn.rect.width / self.scale_factor)
            original_height = int(btn.rect.height / self.scale_factor)
            item = {
                "x": original_x,
                "y": original_y,
                "width": original_width,
                "height": original_height,
                "text": btn.text,
                "text_color": list(btn.text_color),
                "button_color": list(btn.button_color),
                "hover_color": list(btn.hover_color),
                "border_radius": btn.border_radius,
                "callback_id": getattr(btn, "callback_id", "button_clicked"),
                "callback_param": getattr(btn, "callback_param", 0),
            }
            data.append(item)
        with open(self.save_file, "w") as f:
            json.dump(data, f, indent=4)

    def draw_instructions(self, surface):
        """Draw overlay instructions onto the given surface."""
        y_offset = 10
        for line in self.instruction_lines:
            text_surf = self.font.render(line, True, (255, 255, 255))
            surface.blit(text_surf, (10, y_offset))
            y_offset += text_surf.get_height() + 5

    def draw_resize_handles(self, surface):
        """Draw a small square at the bottom-right corner of each button to indicate the resize handle."""
        for btn in self.buttons:
            handle_rect = pygame.Rect(
                btn.rect.right - self.RESIZE_HANDLE_SIZE,
                btn.rect.bottom - self.RESIZE_HANDLE_SIZE,
                self.RESIZE_HANDLE_SIZE,
                self.RESIZE_HANDLE_SIZE,
            )
            pygame.draw.rect(surface, (0, 0, 0), handle_rect)
            pygame.draw.rect(surface, (255, 255, 255), handle_rect, 1)

    def run(self, screen=None, clock=None):
        # Draw the resized background.
        screen.blit(self.background, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Keyboard events.
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_s:
                    self.save_buttons()
                    self.running = False

            # Mouse events.
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if event.button == 1:  # Left-click.
                    # Check for clicks on a resize handle.
                    for btn in reversed(self.buttons):
                        resize_handle_rect = pygame.Rect(
                            btn.rect.right - self.RESIZE_HANDLE_SIZE,
                            btn.rect.bottom - self.RESIZE_HANDLE_SIZE,
                            self.RESIZE_HANDLE_SIZE,
                            self.RESIZE_HANDLE_SIZE,
                        )
                        if resize_handle_rect.collidepoint(pos):
                            self.resizing_button = btn
                            break
                    else:
                        # Not clicking a resize handle: check if clicking on an existing button.
                        for btn in reversed(self.buttons):
                            if btn.rect.collidepoint(pos):
                                self.dragging_button = btn
                                self.drag_offset = (btn.rect.x - pos[0], btn.rect.y - pos[1])
                                break
                        else:
                            # Click did not hit any button: create a new one.
                            new_btn = Button(
                                x=pos[0] - 50,
                                y=pos[1] - 25,
                                width=100,
                                height=50,
                                text="Digital Button",
                                font=self.font,
                                text_color=(255, 255, 255),
                                button_color=(0, 128, 255),
                                hover_color=(0, 102, 204),
                                callback=lambda param=self.button_counter: CALLBACK_MAPPING["button_clicked"](param),
                                border_radius=15
                            )
                            # Save callback info.
                            new_btn.callback_id = "button_clicked"
                            new_btn.callback_param = self.button_counter
                            self.button_counter += 1
                            self.buttons.append(new_btn)
                elif event.button == 3:  # Right-click: remove button.
                    for btn in self.buttons:
                        if btn.rect.collidepoint(pos):
                            self.buttons.remove(btn)
                            break

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_button = None
                    self.resizing_button = None

            elif event.type == pygame.MOUSEMOTION:
                if self.resizing_button is not None:
                    new_width = max(self.MIN_WIDTH, event.pos[0] - self.resizing_button.rect.x)
                    new_height = max(self.MIN_HEIGHT, event.pos[1] - self.resizing_button.rect.y)
                    self.resizing_button.rect.width = new_width
                    self.resizing_button.rect.height = new_height
                elif self.dragging_button is not None:
                    pos = event.pos
                    new_x = pos[0] + self.drag_offset[0]
                    new_y = pos[1] + self.drag_offset[1]
                    self.dragging_button.rect.x = new_x
                    self.dragging_button.rect.y = new_y

        # Draw all digital buttons.
        for btn in self.buttons:
            btn.draw(screen)
        # Draw resize handles.
        self.draw_resize_handles(screen)
        # Draw overlay instructions.
        self.draw_instructions(screen)

        #pygame.display.flip()
        #clock.tick(30)

        return self.running


# -------------------------------
# Example usage:
# -------------------------------
if __name__ == "__main__":
    pygame.init()
    # For demonstration, read an image using cv2.
    # Replace "frame.png" with your actual camera frame file.
    image = cv2.imread("frame.png")
    if image is None:
        raise ValueError("Could not load frame.png. Please provide a valid image file.")

    # Create the editor using the camera frame. The longest side will be scaled to 640 pixels.
    editor = DigitalButtonEditor(background_frame=image, max_display_dimension=640)
    editor.run()
    pygame.quit()
