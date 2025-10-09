# ====== auto_stop.py ======
import os
import threading
import time

def auto_shutdown(delay_seconds=60):
    """
    Stops the instance using OS-level shutdown after delay_seconds.
    Works without IAM/API access - simple and reliable.
    No AWS permissions needed.
    """
    def shutdown():
        print(f"[Auto-Shutdown] Instance will shutdown in {delay_seconds} seconds...")

        # Wait for most of the delay
        if delay_seconds > 10:
            time.sleep(delay_seconds - 10)
            print("[Auto-Shutdown] Instance will stop in 10 seconds...")

        # Countdown warning for last 10 seconds
        countdown = min(10, delay_seconds)
        for i in range(countdown, 0, -1):
            print(f"[Auto-Shutdown] Stopping in {i}...")
            time.sleep(1)

        # Trigger OS shutdown
        print("[Auto-Shutdown] Shutting down now...")
        os.system("sudo shutdown -h now")

    # Run in background so main code can continue
    threading.Thread(target=shutdown, daemon=True).start()
