

import pygame
# The _sdl2 module (available in Pygame 2+) lets us create an extra window.
# If you have problems with this module, consider using a modal preview in your main window.
import pygame._sdl2 as sdl2

from utils.pygame_utils.Button import Button


# -------------------------------------------------------------------------
# Preview window function
# -------------------------------------------------------------------------
def preview_window(buttons_list):
    """
    Open a secondary preview window that shows a frame and allows the user
    to click and drag to select a zone. Once the zone is selected, a new
    Button is created (using the Button class) and appended to buttons_list.
    """
    # Set the preview window size
    preview_width, preview_height = 640, 480

    # Create the preview window using pygame._sdl2 (requires SDL2 and Pygame 2+)
    preview_win = sdl2.Window("Preview", size=(preview_width, preview_height))
    renderer = sdl2.Renderer(preview_win)

    clock = pygame.time.Clock()

    selection_start = None   # Where the mouse was first pressed
    selection_rect = None    # The current rectangle being selected
    selecting = False        # Whether the user is currently dragging a selection

    running = True
    while running:
        for event in pygame.event.get():
            # Close preview if the window is closed
            if event.type == pygame.QUIT:
                running = False

            # Start a new selection when left mouse button is pressed
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    selection_start = event.pos
                    selecting = True

            # Update the rectangle as the mouse moves while dragging
            elif event.type == pygame.MOUSEMOTION:
                if selecting and selection_start:
                    x, y = selection_start
                    current_x, current_y = event.pos
                    # Determine top-left corner and width/height
                    rect_x = min(x, current_x)
                    rect_y = min(y, current_y)
                    rect_width = abs(x - current_x)
                    rect_height = abs(y - current_y)
                    selection_rect = pygame.Rect(rect_x, rect_y, rect_width, rect_height)

            # When the user releases the mouse button, create a new Button
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and selecting:
                    selecting = False
                    if selection_rect and selection_rect.width > 0 and selection_rect.height > 0:
                        # Create a new Button based on the selected rectangle.
                        # (You can customize the text, colors, callback, etc. as needed.)
                        new_button = Button(
                            selection_rect.x,
                            selection_rect.y,
                            selection_rect.width,
                            selection_rect.height,
                            text="New Button",
                            font=pygame.font.SysFont(None, 24),
                            text_color=(255, 255, 255),
                            button_color=(0, 128, 255),
                            hover_color=(0, 255, 255),
                            callback=lambda: print("Button clicked")
                        )
                        buttons_list.append(new_button)
                    # Reset selection variables
                    selection_start = None
                    selection_rect = None

        # --- Drawing in the preview window ---
        renderer.draw_color = (50, 50, 50, 255)  # background color
        renderer.clear()

        # If the user is dragging, draw the selection rectangle in red
        if selection_rect:
            renderer.draw_color = (255, 0, 0, 255)
            renderer.draw_rect(selection_rect)

        renderer.present()
        clock.tick(60)

    preview_win.destroy()

# -------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Main Window")
    clock = pygame.time.Clock()

    # This list will store all the Button instances created
    buttons_list = []
    font = pygame.font.SysFont(None, 36)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Let each button process events (for hover and clicks)
            for btn in buttons_list:
                btn.handle_event(event)

            # If the user presses the 'p' key, open the preview window.
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    preview_window(buttons_list)

        # --- Drawing in the main window ---
        screen.fill((30, 30, 30))

        # Draw all the buttons created (their positions come from the preview)
        for btn in buttons_list:
            btn.draw(screen)

        # Show instructions
        instruction_text = font.render("Press 'P' to open preview and add a button", True, (200, 200, 200))
        screen.blit(instruction_text, (20, 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
