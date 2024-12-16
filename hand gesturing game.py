import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.75,
                       min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils  # For drawing hand landmarks

# Initialize Pygame
pygame.init()
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Gesture-Based Game")

# Set up the player (white box)
player_width = 50
player_height = 50
player_x = window_width // 2 - player_width // 2
player_y = window_height - player_height - 10

# Set up the obstacle (red box)
obstacle_width = 50
obstacle_height = 50
obstacle_x = np.random.randint(0, window_width - obstacle_width)
obstacle_y = 0
obstacle_speed = 3

# Game variables
score = 0
high_score = 0
game_over = False
game_start_time = None

# Fonts for displaying text
font_large = pygame.font.Font('freesansbold.ttf', 64)
font_medium = pygame.font.Font('freesansbold.ttf', 32)

# Game over text
game_over_text = font_large.render('Game Over', True, (255, 255, 255))
restart_button_text = font_medium.render('Restart', True, (255, 255, 255))
restart_button_rect = restart_button_text.get_rect(center=(window_width // 2, window_height // 2 + 50))

# Game loop
running = True
clock = pygame.time.Clock()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Define a variable to store the previous position for smoothing movement
previous_x = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and game_over:
            mouse_pos = pygame.mouse.get_pos()
            if restart_button_rect.collidepoint(mouse_pos):
                # Restart the game
                game_over = False
                score = 0
                player_x = window_width // 2 - player_width // 2
                obstacle_x = np.random.randint(0, window_width - obstacle_width)
                obstacle_y = 0
                game_start_time = time.time()  # Reset game start time

    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # **Convert the BGR image to RGB before processing**
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks for visualization (optional)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the index finger tip landmarks
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Map the index finger's x-coordinate from the webcam frame to the player's x-coordinate in the game window
            index_finger_x = int(index_finger_tip.x * window_width)

            # Implement movement smoothing
            if previous_x is None:
                previous_x = index_finger_x
            else:
                index_finger_x = int(previous_x * 0.8 + index_finger_x * 0.2)  # Smooth movement
                previous_x = index_finger_x  # Update previous_x

            # Move the player based on the hand's x position (responsive movement)
            player_x = index_finger_x - player_width // 2

            # Draw a circle at the index finger's location in the webcam feed for feedback
            index_finger_x_frame = int(index_finger_tip.x * frame_width)
            index_finger_y_frame = int(index_finger_tip.y * frame_height)
            cv2.circle(frame, (index_finger_x_frame, index_finger_y_frame), 10, (0, 255, 0), -1)

    else:
        # If no hand is detected, reset smoothing
        previous_x = None

    # Show the video frame
    cv2.imshow('Gesture Recognition', frame)

    if not game_over:
        # Update the player's position, ensuring the player stays within window boundaries
        player_x = max(0, min(player_x, window_width - player_width))

        # Update the obstacle's position
        obstacle_y += obstacle_speed

        # Check for collision between the player and the obstacle
        if player_x < obstacle_x + obstacle_width and \
                player_x + player_width > obstacle_x and \
                player_y < obstacle_y + obstacle_height and \
                player_y + player_height > obstacle_y:
            # Game over
            game_over = True

        # Reset the obstacle if it goes off the screen
        if obstacle_y > window_height:
            obstacle_x = np.random.randint(0, window_width - obstacle_width)
            obstacle_y = 0
            score += 1  # Increase score for successfully avoiding the obstacle

        # Clear the window
        window.fill((0, 0, 0))

        # Draw the player
        pygame.draw.rect(window, (255, 255, 255), (player_x, player_y, player_width, player_height))

        # Draw the obstacle
        pygame.draw.rect(window, (255, 0, 0), (obstacle_x, obstacle_y, obstacle_width, obstacle_height))

        # Display the score and high score
        score_text = font_medium.render(f'Score: {score}', True, (255, 255, 255))
        high_score_text = font_medium.render(f'High Score: {high_score}', True, (255, 255, 255))
        window.blit(score_text, (10, 10))
        window.blit(high_score_text, (10, 50))

        # Display elapsed time
        if game_start_time is None:
            game_start_time = time.time()  # Initialize start time
        elapsed_time = time.time() - game_start_time
        timer_text = font_medium.render(f'Time: {int(elapsed_time)}s', True, (255, 255, 255))
        window.blit(timer_text, (10, 90))

    # Display "Game Over" text if game over
    if game_over:
        if score > high_score:
            high_score = score  # Update high score if current score is greater
        window.blit(game_over_text, (window_width // 2 - game_over_text.get_width() // 2,
                                     window_height // 2 - game_over_text.get_height() // 2))
        pygame.draw.rect(window, (0, 0, 255), restart_button_rect)
        window.blit(restart_button_text, restart_button_rect)

    # Update the display
    pygame.display.update()

    # Limit the frame rate
    clock.tick(60)

    # Optional: Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()
