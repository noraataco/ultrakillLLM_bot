# utils.py
import win32gui
import win32con
import time
import threading, os
from ctypes import windll

def lock_ultrakill_focus():
    """Focuses the ULTRAKILL window using partial title matching"""
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd) and "ULTRAKILL" in win32gui.GetWindowText(hwnd):
            hwnds.append(hwnd)
        return True
    
    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    if hwnds:
        hwnd = hwnds[0]
        win32gui.SetForegroundWindow(hwnd)
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
        )
        print(f"ULTRAKILL window focused: {win32gui.GetWindowText(hwnd)}")
        return
    print("Warning: ULTRAKILL window not found")

def start_esc_watcher(poll_interval: float = 0.01):
    """
    Spawns a daemon thread that immediately kills the process on Esc press,
    regardless of which window is focused.
    """
    VK_ESC = 0x1B
    user32 = windll.user32

    def _watch():
        while True:
            if user32.GetAsyncKeyState(VK_ESC) & 0x8000:
                os._exit(0)
            time.sleep(poll_interval)

    threading.Thread(target=_watch, daemon=True).start()
