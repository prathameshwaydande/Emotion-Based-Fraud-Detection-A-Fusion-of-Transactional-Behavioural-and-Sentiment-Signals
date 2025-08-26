import time
from pynput import keyboard

keystrokes = []

def on_press(key):
    try:
        press_time = time.time()
        keystrokes.append({"key": key.char, "event": "press", "time": press_time})
    except AttributeError:
        keystrokes.append({"key": str(key), "event": "press", "time": time.time()})

def on_release(key):
    release_time = time.time()
    try:
        keystrokes.append({"key": key.char, "event": "release", "time": release_time})
    except AttributeError:
        keystrokes.append({"key": str(key), "event": "release", "time": release_time})

    # Stop listener with Esc
    if key == keyboard.Key.esc:
        return False

def run_logger():
    print("Start typing. Press ESC to stop.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # Process hold and flight time
    processed = []
    last_release_time = None
    key_down_times = {}

    for stroke in keystrokes:
        key_id = stroke['key']
        event = stroke['event']
        timestamp = stroke['time']

        if event == "press":
            key_down_times[key_id] = timestamp
            if last_release_time is not None:
                flight_time = timestamp - last_release_time
            else:
                flight_time = None
        elif event == "release":
            press_time = key_down_times.get(key_id, None)
            if press_time is not None:
                hold_time = timestamp - press_time
                processed.append({
                    "key": key_id,
                    "hold_time": hold_time,
                    "flight_time": flight_time
                })
                last_release_time = timestamp

    print("\nKeystroke Analysis:")
    for p in processed:
        print(p)

if __name__ == "__main__":
    run_logger()
