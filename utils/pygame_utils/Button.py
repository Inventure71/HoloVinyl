import pygame


class Button:
    def __init__(self, x, y, width, height, text, font, text_color, button_color, hover_color, callback):
        """
        Initialize the button.

        :param x: X-coordinate of the button.
        :param y: Y-coordinate of the button.
        :param width: Width of the button.
        :param height: Height of the button.
        :param text: Text displayed on the button.
        :param font: Pygame font object for the button text.
        :param text_color: Color of the button text.
        :param button_color: Color of the button.
        :param hover_color: Color of the button when hovered.
        :param callback: Function to call when the button is clicked.
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.text_color = text_color
        self.button_color = button_color
        self.hover_color = hover_color
        self.callback = callback
        self.is_hovered = False

    def draw(self, screen):
        """
        Draw the button on the screen.
        :param screen: Pygame surface to draw on.
        """
        # Draw button rectangle
        color = self.hover_color if self.is_hovered else self.button_color
        pygame.draw.rect(screen, color, self.rect)

        # Draw button text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        """
        Handle Pygame events for the button.
        :param event: Pygame event.
        """
        if event.type == pygame.MOUSEMOTION:
            # Check if the mouse is hovering over the button
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.is_hovered:
                # Call the callback function when the button is clicked
                self.callback()

