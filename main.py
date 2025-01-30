import laspy
import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import ttk,filedialog, messagebox

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

    start_viewer_btn = tk.Button(root, text="Start Visualization", command=lambda: start_viewer(pcd1, pcd2))
    start_viewer_btn.pack(pady=20)

    root.mainloop()

def start_viewer(pcd1, pcd2):
    if pcd1 is not None and pcd2 is not None:
        matrix1 = np.eye(4)
        matrix2 = np.eye(4)
        
        translation_offset = np.array([-0.1,0.0,0.0])  
        pcd1.translate(translation_offset)
        
        draw_interactive(pcd1, pcd2, matrix1, matrix2)
    else:
        messagebox.showerror("Error", "Please load both files before starting the viewer.")

def move_cloud(vis, pcd, direction, matrix):
    translation = np.eye(4)
    if direction == "up" :
        step = 0.005
    elif direction == "down" :
        step = 0.005
    else :
        step = 0.01
    
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

def rotate_cloud(vis, pcd):
    rotation = pcd.get_rotation_matrix_from_xyz((0, np.pi / 24, 0))
    pcd.rotate(rotation)
    vis.update_geometry(pcd)

def key_callback_1(vis, pcd1, matrix1):
    vis.register_key_callback(65, lambda vis: move_cloud(vis, pcd1, "left", matrix1))   #A
    vis.register_key_callback(68, lambda vis: move_cloud(vis, pcd1, "right", matrix1))  #D
    vis.register_key_callback(87, lambda vis: move_cloud(vis, pcd1, "up", matrix1))     #W
    vis.register_key_callback(83, lambda vis: move_cloud(vis, pcd1, "down", matrix1))   #S
    vis.register_key_callback(82, lambda vis: rotate_cloud(vis, pcd1))                  #R
    vis.register_key_callback(81, lambda vis: move_cloud(vis, pcd1, "forward", matrix1))  #Q
    vis.register_key_callback(69, lambda vis: move_cloud(vis, pcd1, "backward", matrix1)) #E

def key_callback_2(vis, pcd2, matrix2):
    vis.register_key_callback(74, lambda vis: move_cloud(vis, pcd2, "left", matrix2))   #J
    vis.register_key_callback(76, lambda vis: move_cloud(vis, pcd2, "right", matrix2))  #L
    vis.register_key_callback(73, lambda vis: move_cloud(vis, pcd2, "up", matrix2))     #I
    vis.register_key_callback(75, lambda vis: move_cloud(vis, pcd2, "down", matrix2))   #K
    vis.register_key_callback(85, lambda vis: rotate_cloud(vis, pcd2))                  #U
    vis.register_key_callback(79, lambda vis: move_cloud(vis, pcd2, "forward", matrix2))  #D
    vis.register_key_callback(80, lambda vis: move_cloud(vis, pcd2, "backward", matrix2)) #P

def key_callback_3(vis, pcd1, pcd2, matrix1, matrix2):
    vis.register_key_callback(84, lambda vis: capture_transformation(vis, matrix1))   # กด 'T' เพื่อแสดง Transformation Metric
    vis.register_key_callback(67, lambda vis: register_and_show_error(vis, pcd1, pcd2))  # กด 'C' เพื่อทำ Registration

def capture_transformation(vis,matrix1):
    ctr = vis.get_view_control()
    
    view_matrix = ctr.convert_to_pinhole_camera_parameters().extrinsic

    # สร้างข้อความเพื่อแสดง Transformation Matrix
    matrix_str = "\n".join(["\t".join([f"{val:.1f}" for val in row]) for row in view_matrix])
    
    messagebox.showinfo("Transformation Matrix", 
                        f"Final Transformation Matrix:\n{matrix_str}\n\n"
                        "Matrix Details:\n"
                        "Columns represent the X, Y, Z coordinates and the last column is the translation.\n"
                        "Rows represent the 3D rotation and translation components.")
    
    return False

def register_and_show_error(vis, pcd1, pcd2):
    threshold = 0.005
    reg_icp = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    pcd2.transform(reg_icp.transformation)
    
    # แสดงผลใน MessageBox
    if reg_icp.fitness < threshold:
        messagebox.showerror("Registration Result", "Registration failed due to insufficient points matched.")
    else:
        messagebox.showinfo("Registration Result", f"Registration successful.\nPoints matched: {reg_icp.fitness}")

def draw_interactive(pcd1, pcd2, matrix1, matrix2):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)

    key_callback_1(vis, pcd1, matrix1)  
    key_callback_2(vis, pcd2, matrix2)
    key_callback_3(vis, pcd1, pcd2, matrix1, matrix2)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    create_gui()