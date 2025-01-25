#!/usr/bin/env python3
import argparse
import sys
import os
from datetime import datetime
from subprocess import Popen, PIPE
from collections import Counter
import time
import select
import pygame.mixer
import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Set default values
default_threshold = 10.8
version = '0.8alpha'

# Define a list to hold all signal strengths and detections for visualization
all_signals = []
all_detections = []
detected_frequencies = []

# GPIO Setup
led_pins = [17, 27, 22, 10, 9]  # GPIO pins for the LEDs
GPIO.setmode(GPIO.BCM)  # Use Broadcom pin-numbering scheme
for pin in led_pins:
    GPIO.setup(pin, GPIO.OUT)  # Set each pin as an output

# Define frequency range to sound file mapping
sound_files = {
    (500, 600): 'detection2.ogg',  # Use 'start3.ogg' for frequencies between 500MHz and 600MHz
    (601, 700): 'detection.ogg',  # Use 'detection.ogg' for frequencies between 601MHz and 700MHz
}

def log_message(message, verbose_level=0):
    """Log a message to stdout based on the verbosity level."""
    if args.verbose >= verbose_level:
        print(message)
        sys.stdout.flush()  # Ensure immediate output

def light_up_leds(number_of_hashes):
    """Light up LEDs based on the number of hashes."""
    # Define the number of hashes required to light up one more LED
    hashes_per_led = 3  # This means every 3 hashes light up an additional LED

    # Calculate the number of LEDs to light up based on the number of hashes
    leds_to_light = (number_of_hashes + hashes_per_led - 1) // hashes_per_led  # Ensure rounding up

    # Ensure we do not exceed the number of available LEDs
    leds_to_light = min(leds_to_light, len(led_pins))

    # Turn all LEDs off initially
    GPIO.output(led_pins, GPIO.LOW)
    # print("yo")
    # Turn on the calculated number of LEDs
    for i in range(leds_to_light):
        GPIO.output(led_pins[i], GPIO.HIGH)

    # Keep the LEDs on for some time if you need to visually inspect them
    time.sleep(2)

    # Turn off the LEDs after inspection time
    for pin in led_pins:
        GPIO.output(pin, GPIO.LOW)
    # print("LEDs updated based on hashes.")

def select_sound_for_frequency(freq):
    """Selects the sound file based on the detected frequency."""
    for (start, end), sound_file in sound_files.items():
        if start <= freq <= end:
            return sound_file
    return 'detection.ogg'  # Default sound file

def apply_window(signal):
    """Apply a window function to the signal for better frequency analysis."""
    windowed_signal = np.hamming(len(signal)) * signal  # Using Hamming window function
    return windowed_signal

def perform_fft(signal):
    """Perform Fast Fourier Transform (FFT) on the signal."""
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)
    fft_freq = np.fft.fftfreq(len(fft_result))
    return fft_freq, fft_magnitude

def detect_anomalies(signal):
    """Detect anomalies in signal strength using Isolation Forest."""
    clf = IsolationForest(contamination=0.1)
    clf.fit(signal.reshape(-1, 1))
    anomaly_mask = clf.predict(signal.reshape(-1, 1))
    return anomaly_mask

def process_line(line, threshold, sound):
    """Process one line of input with detailed analysis and formatted output."""
    log_message('Reading', 1)
    try:
        line = line.decode().split(',') if sys.version_info >= (3, 0) else line.split(',')
        time_str, hour, minfreq, maxfreq, step, samples = line[:6]
    except IndexError:
        return False

    log_message(f'Time: {time_str} {hour} MinFreq: {minfreq}, Maxfreq:{maxfreq}, step={step}', 3)

    # Analyze the dbm values
    dbm_line = line[6:]
    max_value, max_pos, detection_dict = float('-inf'), float('-inf'), {}
    for index, value_str in enumerate(dbm_line):
        value = float(value_str)
        if value > max_value:
            max_value, max_pos = value, index
        if value >= threshold:
            detection_dict[index] = value

    detection_freq = float(minfreq) + (float(step) * max_pos)
    formatted_freq = "{:,.3f}".format(detection_freq / 1e6).replace(',', '.')
    int_form = float(formatted_freq)

    hash_marks_length = len(detection_dict)  # Calculate number of hashes once

    if hash_marks_length > 0:
        # This means there's at least one frequency over the threshold, i.e., a detection
        hash_marks = '#' * hash_marks_length
        textline = f'Detection at {formatted_freq} MHz with {hash_marks_length} frequencies over the threshold. {hash_marks}'
        log_message(textline, (0 if args.search else 2))
        if sound:
            selected_sound = select_sound_for_frequency(detection_freq / 1e6)
            pygame.mixer.music.load(selected_sound)
            pygame.mixer.music.play()

        # Light up LEDs based on the number of hashes/detections
        light_up_leds(hash_marks_length)
        detected_frequencies.append(int_form)  # Store the detected frequency

        # Detect anomalies in signal strength
        anomaly_mask = detect_anomalies(np.array(list(detection_dict.values())))
        if -1 in anomaly_mask:
            log_message("Anomaly detected in signal strength!", 2)

    # At the end of the function, add the line's data to the aggregation lists
    all_signals.append(dbm_line)
    all_detections.append(detection_dict)

def visualize_detected_frequencies(detected_frequencies, threshold, start_freq, end_freq):
    plt.figure(figsize=(10, 6))

    # Round the detected frequencies to the nearest whole number
    rounded_freqs = [round(freq) for freq in detected_frequencies]

    # Count the number of detections for each rounded frequency
    frequency_counts = Counter(rounded_freqs)

    # Separate the frequencies and their counts for plotting
    frequencies, counts = zip(*frequency_counts.items())

    # Plot a bar graph
    plt.bar(frequencies, counts, color='r', label='Detections')

    # Plot threshold as a line for reference
    # plt.axhline(y=threshold, color='g', linestyle='--', label='Threshold')

    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Count of Detections')
    plt.title('Detected Frequencies')
    plt.xlim(start_freq, end_freq)
    plt.ylim(0, max(counts) + 1)  # Set the y-axis limit to just above the highest count

    plt.legend()
    plt.grid(axis='y')  # Add a grid on the y-axis for better readability
    plt.show()

def process_file():
    """Read a CSV file created by rtl_power and analyze it offline."""
    if args.verbose > 0:
        log_message(f'Opening file {args.file}', 1)
    try:
        return open(args.file)
    except IOError:
        log_message('No such file.', 0)
        sys.exit(-1)

def process_stdin():
    """Execute the rtl_power tool and get the CSV formatted data."""
    command = f'rtl_power -f {args.startfreq}M:{args.endfreq}M:{args.stepfreq}Khz -g 25 -i 1 -e 86400 -'
    p = Popen(command, shell=True, stdout=PIPE, stderr=open(os.devnull, 'w'), bufsize=1)
    return p.stdout

if _name_ == "_main_":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='CSV file from rtl_power.', action='store', required=False)
    parser.add_argument('-t', '--threshold', help='DBm threshold.', action='store', required=False, type=float, default=default_threshold)
    parser.add_argument('-v', '--verbose', help='Verbose level.', action='store', required=False, type=int, default=0)
    parser.add_argument('-F', '--detfreqthreshold', help='Amount of frequencies over the dbm threshold for detection.', action='store', required=False, type=int, default=1)
    parser.add_argument('-s', '--search', help='Search mode.', action='store_true', required=False, default=True)
    parser.add_argument('-S', '--sound', help='Play sound on detection.', action='store_true', required=False, default=True)
    parser.add_argument('-a', '--startfreq', help='Start frequency for rtl_power.', action='store', type=int, required=False, default=500)
    parser.add_argument('-b', '--endfreq', help='End frequency for rtl_power.', action='store', type=int, required=False, default=700)
    parser.add_argument('-c', '--stepfreq', help='Step frequency for rtl_power.', action='store', type=int, required=False, default=4000)
    args = parser.parse_args()

    log_message(f'Salamandra Hidden Microphone Detector. Version {version}', 0)

    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=8196)
    pygame.mixer.music.load('start3.ogg')
    if args.sound:
        pygame.mixer.music.play()
    time.sleep(1)
    pygame.mixer.music.load('detection.ogg')

    try:
        rfile = process_file() if args.file else process_stdin()
        while True:
            line = rfile.readline()
            if not line:
                break
            process_line(line, args.threshold, args.sound)

    except KeyboardInterrupt:
        log_message('Exiting due to KeyboardInterrupt.', 0)

    finally:
        # Always perform cleanup
        log_message('Cleaning up GPIO and Exiting.', 0)
        GPIO.cleanup()  # Clean up GPIO when exiting

        # Then visualize the data after cleanup, ensuring visualization occurs whether or not there was an interrupt.
        log_message('Visualizing collected data...', 0)
        if detected_frequencies:  # Check to ensure there's data to visualize
            visualize_detected_frequencies(detected_frequencies, default_threshold, 500, 700)
        else:
            log_message('No data collected for visualization.', 0)
