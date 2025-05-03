navigation.py:

import RPi.GPIO as GPIO
import time
import threading  # For real-time obstacle monitoring

# Motor Driver Pins
LEFT_MOTOR_IN1 = 17  
LEFT_MOTOR_IN2 = 18  
RIGHT_MOTOR_IN3 = 23  
RIGHT_MOTOR_IN4 = 22 

# IR Sensor Pins 
IR_CENTER = 19
IR_LEFT = 20
IR_RIGHT = 21

ENA = 5  # PWM Speed Control (Left Motor)
ENB = 6  # PWM Speed Control (Right Motor)

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup([LEFT_MOTOR_IN1, LEFT_MOTOR_IN2, RIGHT_MOTOR_IN3, RIGHT_MOTOR_IN4, ENA, ENB], GPIO.OUT)
GPIO.setup(IR_CENTER, GPIO.IN)
GPIO.setup(IR_LEFT, GPIO.IN)
GPIO.setup(IR_RIGHT, GPIO.IN)

# PWM Setup for Speed Control
pwm_left = GPIO.PWM(ENA, 1000)  # 1kHz frequency
pwm_right = GPIO.PWM(ENB, 1000)
pwm_left.start(50)  # 50% Speed
pwm_right.start(50)
  

# Movement Functions
def move_forward():
    '''global stop_movement
    if not stop_movement:'''
    GPIO.output(LEFT_MOTOR_IN1, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_IN2, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_IN3, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_IN4, GPIO.LOW)
    '''else:
        print("? Obstacle too close! Cannot move forward.")
        stop()'''

def move_backward():
    # Backward movement may be allowed even if an obstacle is ahead.
    GPIO.output(LEFT_MOTOR_IN1, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_IN2, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_IN3, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_IN4, GPIO.HIGH)

def turn_left():
    GPIO.output(LEFT_MOTOR_IN1, GPIO.HIGH)  
    GPIO.output(LEFT_MOTOR_IN2, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_IN3, GPIO.LOW)  
    GPIO.output(RIGHT_MOTOR_IN4, GPIO.HIGH)
    time.sleep(1)
    stop()

def turn_right():
    GPIO.output(LEFT_MOTOR_IN1, GPIO.LOW)  
    GPIO.output(LEFT_MOTOR_IN2, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_IN3, GPIO.HIGH)  
    GPIO.output(RIGHT_MOTOR_IN4, GPIO.LOW)
    time.sleep(1)
    stop()

def stop():
    GPIO.output([LEFT_MOTOR_IN1, LEFT_MOTOR_IN2, RIGHT_MOTOR_IN3, RIGHT_MOTOR_IN4], GPIO.LOW)

# Autonavigation using IR
def auto_navigation():
    center = GPIO.input(IR_CENTER)
    left = GPIO.input(IR_LEFT)
    right = GPIO.input(IR_RIGHT)

    print(f"IR Sensors => Center: {center}, Left: {left}, Right: {right}")

    # Case 1: All clear
    if center == 1 and left == 1 and right == 1:
        print("? Path Clear: Moving Forward")
        move_forward()

    # Case 2: Obstacle directly ahead
    elif center == 0 and left == 1 and right == 1:
        print("?? Obstacle Ahead! Stopping")
        stop()
        time.sleep(0.3)
        print("?? Trying Right Turn")
        turn_right()
        time.sleep(0.3)

    # Case 3: Obstacle ahead and on right -> Turn Left
    elif center == 0 and right == 0 and left == 1:
        print("?? Obstacle Ahead & Right: Turning Left")
        turn_left()
        time.sleep(0.3)

    # Case 4: Obstacle ahead and on left -> Turn Right
    elif center == 0 and left == 0 and right == 1:
        print("?? Obstacle Ahead & Left: Turning Right")
        turn_right()
        time.sleep(0.3)

    # Case 5: Obstacle on all sides
    elif center == 0 and left == 0 and right == 0:
        print("?? Trapped! Reversing")
        move_backward()
        time.sleep(0.5)
        print("?? Attempting Turn")
        turn_right()
        time.sleep(0.3)

    # Case 6: Obstacle only on left
    elif left == 0 and center == 1 and right == 1:
        print("?? Obstacle on Left: Slight Right")
        turn_right()
        time.sleep(0.2)

    # Case 7: Obstacle only on right
    elif right == 0 and center == 1 and left == 1:
        print("?? Obstacle on Right: Slight Left")
        turn_left()
        time.sleep(0.2)

    # Failsafe: Default to Stop
    else:
        print("?? Uncertain state: Stopping")
        stop()



# Control Loop
try:
    while True:
        command = input("Enter command (w=forward, s=backward, a=left, d=right, x=stop, z=auto-navigation, q=exit): ").strip().lower()

        if command == "w":
            print("?? Moving Forward")
            move_forward()
        elif command == "s":
            print("? Moving Backward")
            move_backward()
        elif command == "a":
            print("? Turning Left")
            turn_left()
        elif command == "d":
            print("? Turning Right")
            turn_right()
        elif command == "x":
            print("?? Stopping")
            stop()
        elif command == "z":
            print("?? Switching to AUTO Mode")
            while True:
                auto_navigation()
                time.sleep(0.1)
        elif command == "q":
            print("?? Exiting...")
            break
        else:
            print("? Invalid command! Use w/s/a/d/x/z/q.")

except KeyboardInterrupt:
    print("?? Stopping Robot")

finally:
    stop()
    GPIO.cleanup()