from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.ttk import *
from PIL import Image, ImageTk
import numpy as np
import os
import searches

CANVAS_DIMENSION = 700

class App():
    def __init__(self):
        # Window creation
        self.root = Tk()
        self.root.title("Path Finder - 3000")
        self.root.geometry("1300x900")
        self.root.resizable(True, True)

        # Global state
        self.satellite_image_path = None
        self.mask_image_path = None
        self.sat_image = None
        self.mask_image = None
        self.mask_image_array = None
        self.start_coords = None
        self.end_coords = None

        # Basic Layout
        Label(self.root, text="Path Finder - 3000: Find shortest road path from satellite imaging", font=("Serif", 25)).pack(pady=20)

        # Top menu
        self.menu_frame = Frame(self.root)
        self.menu_frame.pack(pady=20)
        Button(self.menu_frame, text="Select satellite and mask images", command=self.select_files).pack(side=LEFT, padx=20)
        self.algo_combobox = Combobox(self.menu_frame,values=["BFS", "DFS", "A*", "Hill-Climb"])
        self.algo_combobox.pack(side=LEFT, padx=20)
        self.algo_combobox.set("BFS")

        self.toggle_start_end = StringVar()
        Radiobutton(self.menu_frame, text="Path start", variable=self.toggle_start_end, value="start").pack(side=LEFT, padx=20)
        Radiobutton(self.menu_frame, text="Path end", variable=self.toggle_start_end, value="end").pack(side=LEFT,padx=20)

        Button(self.menu_frame, text="Run path search", command=self.run_path_finding).pack(side=LEFT, padx=20)


        # Image canvas
        self.canvas = Canvas(self.root, width=CANVAS_DIMENSION, height=CANVAS_DIMENSION, background="lightgray")
        self.canvas.pack(padx=20, pady=20)
        self.canvas.bind("<Button-1>", self.handle_canvas_click)

        # Begin main loop
        self.root.mainloop()




    def run_path_finding(self):
        if not self.validate_points():
            return
        try:
            match self.algo_combobox.get():
                case "BFS":
                    searches.bfs(
                        (self.start_coords[1], self.start_coords[0]),
                        (self.end_coords[1], self.end_coords[0]),
                        self.satellite_image_path.split(os.sep)[-1].split("_")[0]
                    )
                case "DFS":
                    searches.dfs(
                        (self.start_coords[1], self.start_coords[0]),
                        (self.end_coords[1], self.end_coords[0]),
                        self.satellite_image_path.split(os.sep)[-1].split("_")[0]
                    )
                case "A*":
                    searches.astar(
                        (self.start_coords[1], self.start_coords[0]),
                        (self.end_coords[1], self.end_coords[0]),
                        self.satellite_image_path.split(os.sep)[-1].split("_")[0]
                    )
                case "Hill-Climb":
                    searches.hill_climbing(
                        (self.start_coords[1], self.start_coords[0]),
                        (self.end_coords[1], self.end_coords[0]),
                        self.satellite_image_path.split(os.sep)[-1].split("_")[0]
                    )
        except Exception as e:
            messagebox.showerror("Error", f"No path found by algorithm: {e}")
            raise e

    def handle_canvas_click(self, event):
        if not self.toggle_start_end.get() or not self.satellite_image_path:
            return
        original_x = int((event.x / CANVAS_DIMENSION) * 1024)
        original_y = int((event.y / CANVAS_DIMENSION) * 1024)
        if self.toggle_start_end.get() == "start":
            self.start_coords = (original_x, original_y)
            self.canvas.delete("start-marker")
            marker_size = 4
            self.canvas.create_oval(
                event.x - marker_size, event.y - marker_size,
                event.x + marker_size, event.y + marker_size,
                fill="red", outline="yellow", width=2, tags="start-marker"
            )
            self.canvas.create_text(
                event.x + 8, event.y - 8,
                text="Path start",
                fill="red", font=("Arial", 10, "bold"), tags="start-marker"
            )
        if self.toggle_start_end.get() == "end":
            self.end_coords = (original_x, original_y)
            self.canvas.delete("end-marker")
            marker_size = 4
            self.canvas.create_oval(
                event.x - marker_size, event.y - marker_size,
                event.x + marker_size, event.y + marker_size,
                fill="red", outline="yellow", width=2, tags="end-marker"
            )
            self.canvas.create_text(
                event.x + 8, event.y - 8,
                text="Path end",
                fill="red", font=("Arial", 10, "bold"), tags="end-marker"
            )




    def validate_points(self):
        # Just more readable
        if not self.start_coords or not self.end_coords:
            messagebox.showerror("Error", "Start or End points are not set")
            return False
        sy, sx = self.start_coords
        ey, ex = self.end_coords
        if self.mask_image_array[sx, sy, 0] != 255 or self.mask_image_array[ex, ey, 0] != 255:
            messagebox.showerror("Error", "Points must be on roads")
            return False
        return True



    # Button functionality
    def select_files(self):
        self.satellite_image_path = filedialog.askopenfilename(title="Select satellite image ")
        self.mask_image_path = filedialog.askopenfilename(title="Select mask image")

        if self.satellite_image_path and self.mask_image_path:
            if "sat" not in self.satellite_image_path or "mask" not in self.mask_image_path:
                messagebox.showerror("Error", "Select correct satellite image and mask image")
            self.sat_image = Image.open(self.satellite_image_path)
            self.mask_image = Image.open(self.mask_image_path)
            self.mask_image_array = np.asarray(self.mask_image)
            self.sat_image = self.sat_image.resize((CANVAS_DIMENSION, CANVAS_DIMENSION), Image.Resampling.LANCZOS)
            self.sat_image = ImageTk.PhotoImage(self.sat_image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.sat_image, anchor=NW)
        else:
            messagebox.showerror("Error", "Please select a satellite image and mask image")



# Basic window creation





App()
