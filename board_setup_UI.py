import pygame

from pygame_utils.Button import Button
from pygame_utils.TextField import TextField


def button_clicked_1():
    print("Button 1!")

def button_clicked_2():
    print("Button 2!")


class Board_Setup_UI:
    def __init__(self):
        self.screen = pygame.display.set_mode((1024, 600))
        pygame.display.set_caption("Board Setup")
        self.clock = pygame.time.Clock()
        self.running = True

        self.width = 1024
        self.height = 600
        self.font = pygame.font.Font(None, 36)

        self.buttons = [
            Button(
                x=self.width // 2 - 100,  # Centered horizontally
                y=self.height // 2 - 50,  # Position it visibly
                width=200,
                height=50,
                text="Start",
                font=self.font,
                text_color=(255, 255, 255),
                button_color=(0, 128, 255),
                hover_color=(0, 102, 204),
                callback=button_clicked_1,
            ),
            Button(
                x=self.width // 2 - 100,  # Centered horizontally
                y=self.height // 2 + 60,  # Below the first button
                width=200,
                height=50,
                text="Exit",
                font=self.font,
                text_color=(255, 255, 255),
                button_color=(0, 128, 255),
                hover_color=(0, 102, 204),
                callback=button_clicked_2,
            ),
        ]

    def run(self):
        while self.running:
            #self.screen.fill((0, 0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                for button in self.buttons:
                    button.handle_event(event)


            for button in self.buttons:
                button.draw(self.screen)

            self.clock.tick(30)
            pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    pygame.init()
    board_setup = Board_Setup_UI()
    board_setup.run()