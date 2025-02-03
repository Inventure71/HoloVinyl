import os
import json
import cv2
import pygame
from pygame.locals import *
from utils.pygame_utils.Button import Button  # Adjust the import as needed

class DigitalButtonEditor:
    # Constants for resizing buttons
    RESIZE_HANDLE_SIZE = 10
    MIN_WIDTH = 30
    MIN_HEIGHT = 30

    def __init__(self, background_frame, callback, save_file="variables/digital_buttons.json", font=None):
        """
        :param background_frame: A camera frame, either as a cv2 image (numpy array) or a pygame.Surface.
        :param save_file: File where digital button configurations are saved.
        :param font: Pygame font to use. If None, a default font is created.
        """
        # Map callback identifier strings to functions.
        self.CALLBACK_MAPPING = {
            "button_clicked": callback,
        }

        self.save_file = save_file
        self.font = font or pygame.font.Font(None, 32)
        self.buttons = []
        self.running = True

        # For moving buttons:
        self.dragging_button = None
        self.drag_offset_global = (0, 0)

        # For resizing buttons:
        self.resizing_button = None
        self.resize_start_global = (0, 0)
        self.resize_start_width = 0
        self.resize_start_height = 0

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
        if isinstance(background_frame, pygame.Surface):
            self.original_surface = background_frame.copy()
        else:
            # Assume cv2 frame (BGR); convert to RGB.
            background_rgb = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
            self.original_surface = pygame.image.frombuffer(
                background_rgb.tobytes(),
                (background_rgb.shape[1], background_rgb.shape[0]),
                "RGB"
            )
        self.original_width, self.original_height = self.original_surface.get_size()

        # Load any previously saved buttons into self.buttons (global coordinates).
        self.load_buttons()

    def load_buttons(self):
        """Load digital buttons from the save file (if it exists) in global coordinates."""
        if os.path.exists(self.save_file):
            with open(self.save_file, "r") as f:
                data = json.load(f)
            self.buttons = []
            for item in data:
                btn = Button(
                    x=item.get("x", 0),
                    y=item.get("y", 0),
                    width=item.get("width", 100),
                    height=item.get("height", 50),
                    text=item.get("text", "Digital Button"),
                    font=self.font,
                    text_color=tuple(item.get("text_color", [255, 255, 255])),
                    button_color=tuple(item.get("button_color", [0, 128, 255])),
                    hover_color=tuple(item.get("hover_color", [0, 102, 204])),
                    callback=lambda param=item.get("callback_param", 0), cid=item.get("callback_id", "button_clicked"): self.CALLBACK_MAPPING[cid](param),
                    border_radius=item.get("border_radius", 15)
                )
                btn.callback_id = item.get("callback_id", "button_clicked")
                btn.callback_param = item.get("callback_param", 0)
                self.buttons.append(btn)
                if btn.callback_param >= self.button_counter:
                    self.button_counter = btn.callback_param + 1
        else:
            self.buttons = []

    def save_buttons(self):
        """Save the current digital buttons in global coordinates."""
        data = []
        for btn in self.buttons:
            item = {
                "x": btn.rect.x,
                "y": btn.rect.y,
                "width": btn.rect.width,
                "height": btn.rect.height,
                "text": btn.text,
                "text_color": list(btn.text_color),
                "button_color": list(btn.button_color),
                "hover_color": list(btn.hover_color),
                "border_radius": btn.border_radius,
                "callback_id": btn.callback_id,
                "callback_param": btn.callback_param,
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

    def run(self, screen, clock):
        # Get current screen dimensions
        screen_width, screen_height = screen.get_size()

        # Calculate scaling to fit the original image into the current screen while maintaining aspect ratio
        scale_width = screen_width / self.original_width
        scale_height = screen_height / self.original_height
        scale_factor = min(scale_width, scale_height)
        scaled_width = int(self.original_width * scale_factor)
        scaled_height = int(self.original_height * scale_factor)

        # Scale the background
        scaled_background = pygame.transform.scale(self.original_surface, (scaled_width, scaled_height))

        # Calculate offset to center the background
        offset_x = (screen_width - scaled_width) // 2
        offset_y = (screen_height - scaled_height) // 2

        # Draw the scaled background centered
        screen.blit(scaled_background, (offset_x, offset_y))

        # Process events
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_s:
                    self.save_buttons()
                    self.running = False
            elif event.type == MOUSEBUTTONDOWN:
                screen_mouse_pos = pygame.mouse.get_pos()
                # Convert to editor surface coordinates
                editor_mouse_pos = (
                    screen_mouse_pos[0] - offset_x,
                    screen_mouse_pos[1] - offset_y
                )
                if (0 <= editor_mouse_pos[0] < scaled_width and
                    0 <= editor_mouse_pos[1] < scaled_height):
                    # Convert to global coordinates
                    global_mouse_pos = (
                        editor_mouse_pos[0] / scale_factor,
                        editor_mouse_pos[1] / scale_factor
                    )
                    if event.button == 1:  # Left-click
                        # Check resize handles
                        for btn in reversed(self.buttons):
                            btn_display_rect = pygame.Rect(
                                btn.rect.x * scale_factor,
                                btn.rect.y * scale_factor,
                                btn.rect.width * scale_factor,
                                btn.rect.height * scale_factor
                            )
                            resize_handle_rect = pygame.Rect(
                                btn_display_rect.right - self.RESIZE_HANDLE_SIZE,
                                btn_display_rect.bottom - self.RESIZE_HANDLE_SIZE,
                                self.RESIZE_HANDLE_SIZE,
                                self.RESIZE_HANDLE_SIZE
                            )
                            # Check collision in display coordinates
                            if resize_handle_rect.collidepoint(editor_mouse_pos):
                                self.resizing_button = btn
                                self.resize_start_global = global_mouse_pos
                                self.resize_start_width = btn.rect.width
                                self.resize_start_height = btn.rect.height
                                break
                        else:
                            # Check button click
                            for btn in reversed(self.buttons):
                                btn_display_rect = pygame.Rect(
                                    btn.rect.x * scale_factor,
                                    btn.rect.y * scale_factor,
                                    btn.rect.width * scale_factor,
                                    btn.rect.height * scale_factor
                                )
                                if btn_display_rect.collidepoint(editor_mouse_pos):
                                    self.dragging_button = btn
                                    self.drag_offset_global = (
                                        btn.rect.x - global_mouse_pos[0],
                                        btn.rect.y - global_mouse_pos[1]
                                    )
                                    break
                            else:
                                # Add new button
                                new_btn = Button(
                                    x=global_mouse_pos[0] - 50,  # Centered
                                    y=global_mouse_pos[1] - 25,
                                    width=100,
                                    height=50,
                                    text="Digital Button",
                                    font=self.font,
                                    text_color=(255, 255, 255),
                                    button_color=(0, 128, 255),
                                    hover_color=(0, 102, 204),
                                    callback=lambda param=self.button_counter: self.CALLBACK_MAPPING["button_clicked"](param),
                                    border_radius=15
                                )
                                new_btn.callback_id = "button_clicked"
                                new_btn.callback_param = self.button_counter
                                self.button_counter += 1
                                self.buttons.append(new_btn)
                    elif event.button == 3:  # Right-click
                        # Remove button
                        for btn in reversed(self.buttons):
                            btn_display_rect = pygame.Rect(
                                btn.rect.x * scale_factor,
                                btn.rect.y * scale_factor,
                                btn.rect.width * scale_factor,
                                btn.rect.height * scale_factor
                            )
                            if btn_display_rect.collidepoint(editor_mouse_pos):
                                self.buttons.remove(btn)
                                break
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_button = None
                    self.resizing_button = None
            elif event.type == MOUSEMOTION:
                if self.dragging_button is not None:
                    screen_mouse_pos = pygame.mouse.get_pos()
                    editor_mouse_pos = (
                        screen_mouse_pos[0] - offset_x,
                        screen_mouse_pos[1] - offset_y
                    )
                    global_mouse_pos = (
                        editor_mouse_pos[0] / scale_factor,
                        editor_mouse_pos[1] / scale_factor
                    )
                    self.dragging_button.rect.x = global_mouse_pos[0] + self.drag_offset_global[0]
                    self.dragging_button.rect.y = global_mouse_pos[1] + self.drag_offset_global[1]
                elif self.resizing_button is not None:
                    screen_mouse_pos = pygame.mouse.get_pos()
                    editor_mouse_pos = (
                        screen_mouse_pos[0] - offset_x,
                        screen_mouse_pos[1] - offset_y
                    )
                    global_mouse_pos = (
                        editor_mouse_pos[0] / scale_factor,
                        editor_mouse_pos[1] / scale_factor
                    )
                    delta_x = global_mouse_pos[0] - self.resize_start_global[0]
                    delta_y = global_mouse_pos[1] - self.resize_start_global[1]
                    new_width = max(self.MIN_WIDTH, self.resize_start_width + delta_x)
                    new_height = max(self.MIN_HEIGHT, self.resize_start_height + delta_y)
                    self.resizing_button.rect.width = new_width
                    self.resizing_button.rect.height = new_height

        # Draw buttons
        for btn in self.buttons:
            # Calculate display coordinates
            display_x = int(btn.rect.x * scale_factor) + offset_x
            display_y = int(btn.rect.y * scale_factor) + offset_y
            display_width = int(btn.rect.width * scale_factor)
            display_height = int(btn.rect.height * scale_factor)
            temp_btn = Button(
                x=display_x,
                y=display_y,
                width=display_width,
                height=display_height,
                text=btn.text,
                font=self.font,
                text_color=btn.text_color,
                button_color=btn.button_color,
                hover_color=btn.hover_color,
                callback=btn.callback,
                border_radius=btn.border_radius
            )
            temp_btn.draw(screen)
            # Draw resize handle
            handle_rect = pygame.Rect(
                display_x + display_width - self.RESIZE_HANDLE_SIZE,
                display_y + display_height - self.RESIZE_HANDLE_SIZE,
                self.RESIZE_HANDLE_SIZE,
                self.RESIZE_HANDLE_SIZE
            )
            pygame.draw.rect(screen, (0, 0, 0), handle_rect)
            pygame.draw.rect(screen, (255, 255, 255), handle_rect, 1)

        # Draw instructions
        self.draw_instructions(screen)

        return self.running, self.buttons

# -------------------------------
# Example usage:
# -------------------------------
if __name__ == "__main__":
    pygame.init()
    image = cv2.imread("frame.png")
    if image is None:
        raise ValueError("Could not load frame.png. Please provide a valid image file.")

    screen = pygame.display.set_mode((800, 600), RESIZABLE)
    clock = pygame.time.Clock()

    editor = DigitalButtonEditor(background_frame=image)

    running = True
    while running:
        screen.fill((0, 0, 0))  # Clear screen with black background
        running, _ = editor.run(screen, clock)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()