import laspy
import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def read_las_file(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return las, pcd

def las2Array(lasdata):
    points = np.vstack((lasdata.x, lasdata.y, lasdata.z)).transpose()
    
    if hasattr(lasdata, 'red') and hasattr(lasdata, 'green') and hasattr(lasdata, 'blue'):
        colors = np.vstack((lasdata.red, lasdata.green, lasdata.blue)).transpose()
        colors = colors / 65535.0 
    else:
        colors = np.ones((len(points), 3))
    
    return points, colors

def pickfile(sot):
    file_path = filedialog.askopenfilename(title=f"Select a {sot} file", filetypes=[("Las file", "*.las")])
    if file_path:
        print(f"Selected file: {file_path}")
        return file_path
    else:
        print("No file selected")
        return None

def create_gui():
    root = tk.Tk()
    root.title("Point Cloud Program")
    root.geometry("400x300")
    root.configure(bg='#F0F0F0')
    
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    style.configure("TLabel", font=("Helvetica", 12), background='#F0F0F0')

    pcd1 = None
    pcd2 = None
    step_value = tk.DoubleVar(value=0.005)

    def load_file1():
        nonlocal pcd1  
        file_path = pickfile("source 1")
        if file_path:
            try:
                las_data, pcd = read_las_file(file_path)
                points, colors = las2Array(las_data)
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd1 = pcd  
                messagebox.showinfo("Success", "File 1 loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def load_file2():
        nonlocal pcd2  
        file_path = pickfile("source 2")
        if file_path:
            try:
                las_data, pcd = read_las_file(file_path)
                points, colors = las2Array(las_data)
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd2 = pcd  
                messagebox.showinfo("Success", "File 2 loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    load_btn1 = tk.Button(root, text="Load Source 1", command=load_file1)
    load_btn1.pack(pady=10)

    load_btn2 = tk.Button(root, text="Load Source 2", command=load_file2)
    load_btn2.pack(pady=10)

    step_label = tk.Label(root, text="Step value:")
    step_label.pack(pady=5)

    step_entry = tk.Entry(root, textvariable=step_value)
    step_entry.pack(pady=5)

    start_viewer_btn = tk.Button(root, text="Start Visualization", command=lambda: start_viewer(pcd1, pcd2, step_value.get()))
    start_viewer_btn.pack(pady=20)

    root.mainloop()

def start_viewer(pcd1, pcd2, step):
    if pcd1 is not None and pcd2 is not None:
        matrix1 = np.eye(4)
        matrix2 = np.eye(4)
        
        translation_offset = np.array([-0.1,0.0,0.0])  
        pcd1.translate(translation_offset)
        
        draw_interactive(pcd1, pcd2, matrix1, matrix2, step)
    else:
        messagebox.showerror("Error", "Please load both files before starting the viewer.")

def move_cloud(vis, pcd, direction, matrix, step):
    translation = np.eye(4)
    
    if direction == "left":
        matrix[0, 3] -= step
        translation[0, 3] -= step
    elif direction == "right":
        matrix[0, 3] += step
        translation[0, 3] += step
    elif direction == "up":
        matrix[1, 3] += step
        translation[1, 3] += step
    elif direction == "down":
        matrix[1, 3] -= step
        translation[1, 3] -= step
    elif direction == "forward":
        matrix[2, 3] += step
        translation[2, 3] += step
    elif direction == "backward":
        matrix[2, 3] -= step
        translation[2, 3] -= step

    pcd.transform(translation)
    vis.update_geometry(pcd)

def rotate_cloud(vis, pcd, axis="x"):
    if axis == "x":
        axis_values = (np.pi / 24, 0, 0)
    elif axis == "y":
        axis_values = (0, np.pi / 24, 0)
    elif axis == "z":
        axis_values = (0, 0, np.pi / 24)

    rotation = pcd.get_rotation_matrix_from_xyz(axis_values)
    pcd.rotate(rotation)
    vis.update_geometry(pcd)

def key_callback_1(vis, pcd1, matrix1, step):
    vis.register_key_callback(65, lambda vis: move_cloud(vis, pcd1, "left", matrix1, step))     #A
    vis.register_key_callback(68, lambda vis: move_cloud(vis, pcd1, "right", matrix1, step))    #D
    vis.register_key_callback(87, lambda vis: move_cloud(vis, pcd1, "up", matrix1, step))       #W
    vis.register_key_callback(83, lambda vis: move_cloud(vis, pcd1, "down", matrix1, step))     #S
    vis.register_key_callback(88, lambda vis: rotate_cloud(vis, pcd1, "x"))                     #X
    vis.register_key_callback(89, lambda vis: rotate_cloud(vis, pcd1, "y"))                     #Y
    vis.register_key_callback(90, lambda vis: rotate_cloud(vis, pcd1, "z"))                     #Z
    vis.register_key_callback(88, lambda vis: rotate_cloud(vis, pcd1))                          #X
    vis.register_key_callback(89, lambda vis: rotate_cloud(vis, pcd1))                          #Y
    vis.register_key_callback(90, lambda vis: rotate_cloud(vis, pcd1))                          #Z
    vis.register_key_callback(81, lambda vis: move_cloud(vis, pcd1, "forward", matrix1, step))  #Q
    vis.register_key_callback(69, lambda vis: move_cloud(vis, pcd1, "backward", matrix1, step)) #E

# def key_callback_1(vis, pcd1, matrix1, step):
#     vis.register_key_callback(263, lambda vis: move_cloud(vis, pcd1, "left", matrix1, step))     #left arrow
#     vis.register_key_callback(262, lambda vis: move_cloud(vis, pcd1, "right", matrix1, step))    #right arrow
#     vis.register_key_callback(265, lambda vis: move_cloud(vis, pcd1, "up", matrix1, step))       #up arrow
#     vis.register_key_callback(264, lambda vis: move_cloud(vis, pcd1, "down", matrix1, step))     #down arrow
#     vis.register_key_callback(88, lambda vis: rotate_cloud(vis, pcd1, "x"))                          #X
#     vis.register_key_callback(89, lambda vis: rotate_cloud(vis, pcd1, "y"))                          #Y
#     vis.register_key_callback(90, lambda vis: rotate_cloud(vis, pcd1, "z"))                          #Z
#     vis.register_key_callback(81, lambda vis: move_cloud(vis, pcd1, "forward", matrix1, step))  #Q
#     vis.register_key_callback(69, lambda vis: move_cloud(vis, pcd1, "backward", matrix1, step)) #E

def key_callback_2(vis, pcd2, matrix2, step):
    vis.register_key_callback(74, lambda vis: move_cloud(vis, pcd2, "left", matrix2, step))     #J
    vis.register_key_callback(76, lambda vis: move_cloud(vis, pcd2, "right", matrix2, step))    #L
    vis.register_key_callback(73, lambda vis: move_cloud(vis, pcd2, "up", matrix2, step))       #I
    vis.register_key_callback(75, lambda vis: move_cloud(vis, pcd2, "down", matrix2, step))     #K
    vis.register_key_callback(85, lambda vis: rotate_cloud(vis, pcd2))                          #U
    vis.register_key_callback(79, lambda vis: move_cloud(vis, pcd2, "forward", matrix2, step))  #O
    vis.register_key_callback(80, lambda vis: move_cloud(vis, pcd2, "backward", matrix2, step)) #P

def draw_interactive(pcd1, pcd2, matrix1, matrix2, step):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)

    key_callback_1(vis, pcd1, matrix1, step)
    key_callback_2(vis, pcd2, matrix2, step)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    create_gui()
