import pygame


class TextField:
    def __init__(self, x, y, width, height, font, text_color, bg_color, border_color, border_width=2):
        """
        Initialize the text field.

        :param x: X-coordinate of the text field.
        :param y: Y-coordinate of the text field.
        :param width: Width of the text field.
        :param height: Height of the text field.
        :param font: Pygame font object for the text.
        :param text_color: Color of the text.
        :param bg_color: Background color of the text field.
        :param border_color: Border color of the text field.
        :param border_width: Border width of the text field.
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.text_color = text_color
        self.bg_color = bg_color
        self.border_color = border_color
        self.border_width = border_width
        self.text = ""  # The text entered by the user
        self.active = False  # Whether the text field is focused
        self.cursor_visible = True
        self.cursor_timer = 0

    def handle_event(self, event):
        """
        Handle Pygame events for the text field.
        :param event: Pygame event.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle active state based on whether the user clicks inside the text field
            self.active = self.rect.collidepoint(event.pos)

        if event.type == pygame.KEYDOWN and self.active:
            # Handle copy-paste functionality
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_CTRL:
                if event.key == pygame.K_c:  # Copy
                    pygame.scrap.put(pygame.SCRAP_TEXT, self.text.encode('utf-8'))
                    print(f"Copied to clipboard: {self.text}")
                elif event.key == pygame.K_v:  # Paste
                    clipboard_text = pygame.scrap.get(pygame.SCRAP_TEXT)
                    if clipboard_text:
                        # Decode clipboard text, remove null characters, and append
                        sanitized_text = clipboard_text.decode('utf-8').replace('\x00', '')
                        self.text += sanitized_text
                        print(f"Pasted from clipboard: {sanitized_text}")
            else:
                # Handle other key inputs
                if event.key == pygame.K_BACKSPACE:
                    # Remove the last character
                    self.text = self.text[:-1]
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    # Handle 'Enter' key press (you can define behavior for it here)
                    self.active = False
                else:
                    # Add the pressed key to the text (if valid)
                    self.text += event.unicode

    def update(self):
        """
        Update the text field (e.g., for cursor blinking).
        """
        if self.active:
            self.cursor_timer += 1
            if self.cursor_timer >= 30:  # Blink every half second
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0
        else:
            self.cursor_visible = False

    def draw(self, screen):
        """
        Draw the text field on the screen.
        :param screen: Pygame surface to draw on.
        """
        # Draw the background and border
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, self.border_width)

        # Render the text
        text_surface = self.font.render(self.text, True, self.text_color)
        screen.blit(text_surface, (self.rect.x + 5, self.rect.y + (self.rect.height - text_surface.get_height()) // 2))

        # Draw the blinking cursor if the field is active
        if self.active and self.cursor_visible:
            cursor_x = self.rect.x + 5 + text_surface.get_width()
            cursor_y = self.rect.y + 5
            cursor_height = self.rect.height - 10
            pygame.draw.rect(screen, self.text_color, (cursor_x, cursor_y, 2, cursor_height))