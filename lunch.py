import tkinter as tk
import threading
import subprocess
import tkinter.scrolledtext as st

class EmotionDetectionLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection Launcher")

        self.init_gui()

    def init_gui(self):
        tk.Label(self.root, text="Select an Emotion Detection Mode:").grid(row=0, column=0, columnspan=2)

        tk.Button(self.root, text="Offline Detection", command=self.run_offline_detection).grid(row=1, column=0)
        tk.Button(self.root, text="Live Detection", command=self.run_live_detection).grid(row=1, column=1)

        self.output_text = st.ScrolledText(self.root, wrap=tk.WORD, width=50, height=15)
        self.output_text.grid(row=2, column=0, columnspan=2, pady=10, padx=10)

    def run_offline_detection(self):
        threading.Thread(target=self.run_script, args=("offline.py",)).start()

    def run_live_detection(self):
        threading.Thread(target=self.run_script, args=("live.py",)).start()

    def run_script(self, script_name):
        try:
            result = subprocess.run(["python", script_name], check=True, capture_output=True, text=True)
            self.display_output(result.stdout)
        except subprocess.CalledProcessError as e:
            self.display_output(e.stdout)
            self.display_output(e.stderr)

    def display_output(self, output):
        self.output_text.insert(tk.END, output)
        self.output_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionLauncherApp(root)
    root.mainloop()
