import pygame
import time

# Initialize pygame and joystick
pygame.init()
pygame.joystick.init()

# Check for joysticks
if pygame.joystick.get_count() == 0:
    print("No joystick found!")
    exit()
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick initialized: {joystick.get_name()}")

# Mapping info
num_axes = joystick.get_numaxes()
num_buttons = joystick.get_numbuttons()

print(f"Axes: {num_axes}, Buttons: {num_buttons}")

try:
    while True:
        pygame.event.pump()

        axes = [joystick.get_axis(i) for i in range(num_axes)]
        buttons = [joystick.get_button(i) for i in range(num_buttons)]

        print("Axes:", ["{:.2f}".format(a) for a in axes])
        print("Buttons:", buttons)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting.")
finally:
    pygame.quit()
