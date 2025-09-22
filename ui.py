from tkinter import *
from tkinter import filedialog, messagebox

# Basic window creation
root = Tk()
root.title("Path Finder - 3000")
root.geometry("1300x900")
root.resizable(True, True)

# Global state
satellite_image_path = None
mask_image_path = None

# Layout start
Label(root, text="Path Finder - 3000: Find shortest road path from satellite imaging", font=("Serif", 25)).pack(pady=20)

def select_files():
    global satellite_image_path
    global mask_image_path
    satellite_image_path = filedialog.askopenfilename(title = "Select satellite image ")
    mask_image_path = filedialog.askopenfilename(title = "Select mask image")

    if satellite_image_path and mask_image_path:
        if "sat" not in satellite_image_path or "mask" not in mask_image_path:
            messagebox.showerror("Error", "Select correct satellite image and mask image")
    else:
        messagebox.showerror("Error", "Please select a satellite image and mask image")

Button(root, text="Select satellite and mask images", command=select_files).pack(pady=20)





root.mainloop()