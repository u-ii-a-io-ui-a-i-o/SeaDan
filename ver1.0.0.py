import laspy
import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from calculate_volume import calculate_grid_median_with_kdtree
import multiprocessing
import time
import matplotlib.pyplot as plt

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
    root.iconbitmap('./image/beach_3586.ico')
    root.title("SeaDan")
    root.geometry("1000x1000")
    root.configure(bg='#E1F5FE')

    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10, background="#42A5F5", foreground="white")
    style.configure("TButton:hover", background="#1E88E5")
    style.configure("TLabel", font=("Helvetica", 12), background='#E1F5FE', foreground="#0277BD")
    
    header_label = tk.Label(root, text="Welcome to SeaDan", font=("Helvetica", 16, 'bold'), bg='#E1F5FE', fg="#0277BD")
    header_label.grid(row=0, column=0, columnspan=2, pady=10, padx=20, sticky="n")
    
    pcd1 = None
    pcd2 = None
    step_value = tk.DoubleVar(value=0.05)

    def load_file1():
        nonlocal pcd1  
        file_path = pickfile("source")
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
        file_path = pickfile("target")
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
    
    load_btn1 = tk.Button(root, text="Load Source", command=load_file1, width=20, height=2)
    load_btn1.grid(row=1, column=0 ,columnspan=2, pady=10, padx=20, sticky="n")

    load_btn2 = tk.Button(root, text="Load Target", command=load_file2, width=20, height=2)
    load_btn2.grid(row=2, column=0 ,columnspan=2, pady=10, padx=20, sticky="n")

    step_label = tk.Label(root, text="Step value", font=("Helvetica", 12), bg="#D3D3D3", fg="#0277BD")
    step_label.grid(row=3,pady=10, padx=5)

    step_entry = tk.Entry(root, textvariable=step_value)
    step_entry.grid(row=4 , pady=10, padx=5)

    start_vis_source = tk.Button(root, text="Start Visualization Source", command=lambda: visualize_a_point_cloud(pcd1, "source"),width=22, height=2)
    start_vis_target = tk.Button(root, text="Start Visualization Target", command=lambda: visualize_a_point_cloud(pcd2, "target"),width=22, height=2)
    start_vis_source.grid(row=5,pady=10, padx=5)
    start_vis_target.grid(row=6,pady=10, padx=5)

    start_viewer_btn = tk.Button(root, text="Start visualize & align", command=lambda: start_viewer(pcd1, pcd2, step_value.get()),width=22, height=2)
    start_viewer_btn.grid(row=7,pady=10, padx=5)

    start_calculate_btn = tk.Button(root, text="Start Calculate Volume", command=lambda: start_calculate(pcd1, pcd2),width=22, height=2)
    start_calculate_btn.grid(row=8,pady=10, padx=5)
    
    # Add a label for displaying total volume and elapsed time
    result_label = tk.Label(root, text="Results will appear here.", font=("Helvetica", 14), bg='#E1F5FE', fg="#0277BD")
    result_label.grid(row=9, column=0, columnspan=2, pady=20, padx=20)

    man_btn = tk.Button(root, text="Help", command=show_man,width=8, height=2)
    man_btn.grid(row=10, column=0 ,columnspan=2, pady=10, padx=20, sticky="n")
    
    def logout():
        response = messagebox.askyesno("Logout", "Are you sure you want to log out?")
        if response:
            root.destroy() 
            login_window()  
    logout_btn = tk.Button(root, text="Logout", command=logout, width=20, height=2)
    logout_btn.grid(row=11, column=0 ,columnspan=2, pady=10, padx=20, sticky="n")
    
    text_frame = tk.Frame(root, width=270, height=350, relief="sunken", borderwidth=2)
    text_frame.grid(row=4, column=1, rowspan=5, padx=5, pady=10, sticky="n")
    text_box = tk.Text(text_frame, wrap="word", width=30, height=25)
    text_box.insert("1.0", "Step Value Normal = 0.05 (ใช้เมื่อไฟล์มีขนาด มากกว่าKB แต่น้อยกว่า10KB) ถ้ามากกว่่านั้นให้เพิ่มจำนวน Step Value ตามความเหมาะสม")
    text_box.pack(expand=True, fill="both")

    def start_calculate(src_pcd, target_pcd):
        src_points = np.asarray(src_pcd.points)
        tgt_points = np.asarray(target_pcd.points)

    # Bounding box and grid setup
        src_x_min, src_x_max = min(src_points[:, 0]), max(src_points[:, 0])
        src_y_min, src_y_max = min(src_points[:, 1]), max(src_points[:, 1])
        tgt_x_min, tgt_x_max = min(tgt_points[:, 0]), max(tgt_points[:, 0])
        tgt_y_min, tgt_y_max = min(tgt_points[:, 1]), max(tgt_points[:, 1])

        gbl_x_min, gbl_x_max = min(src_x_min, tgt_x_min), max(src_x_max, tgt_x_max)
        gbl_y_min, gbl_y_max = min(src_y_min, tgt_y_min), max(src_y_max, tgt_y_max)

        col_width = 1
        row_width = 1

        n_cols = math.ceil((gbl_x_max - gbl_x_min) / col_width)
        n_rows = math.ceil((gbl_y_max - gbl_y_min) / row_width)

        start_time = time.perf_counter()

    # Prepare argument tuples for multiprocessing
        src_args = (src_points, gbl_x_min, gbl_y_min, col_width, row_width, n_rows, n_cols)
        tgt_args = (tgt_points, gbl_x_min, gbl_y_min, col_width, row_width, n_rows, n_cols)

        if n_cols * n_rows < 1:
        # Use multiprocessing to calculate median altitudes for both source and target
            with multiprocessing.Pool(processes=4) as pool:
            # starmap will unpack each tuple argument list correctly
                results = pool.starmap(calculate_grid_median_with_kdtree, [src_args, tgt_args])
        
        # Unpack results
            src_avg_alt_mat, count_point_src = results[0]
            tgt_avg_alt_mat, count_point_tgt = results[1]
        
        else:
            src_avg_alt_mat, count_point_src = calculate_grid_median_with_kdtree(src_points, gbl_x_min, gbl_y_min, col_width, row_width, n_rows, n_cols)
            tgt_avg_alt_mat, count_point_tgt = calculate_grid_median_with_kdtree(tgt_points, gbl_x_min, gbl_y_min, col_width, row_width, n_rows, n_cols)

        print(count_point_src)
        print(count_point_tgt)

    # Delta altitude matrix and volume change calculation
        delta_alt_mat = tgt_avg_alt_mat - src_avg_alt_mat
        cell_area = col_width * row_width
        volume_change = (np.array(tgt_avg_alt_mat) - np.array(src_avg_alt_mat)) * cell_area

        volume_change = np.nan_to_num(volume_change)
        total_volume_change = np.sum(volume_change)
    
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Hide unused subplot (row 2, column 2)
        axes[1, 1].axis('off')

    # Subplot 1: Average Altitude (Source)
        im1 = axes[0, 0].imshow(src_avg_alt_mat,cmap='viridis')
        axes[0, 0].set_title('Average Altitude (Source)')
        axes[0, 0].set_xlabel(f'{n_cols} columns')
        axes[0, 0].set_ylabel(f'{n_rows} rows')

    # Subplot 2: Average Altitude (Target)
        im2 = axes[0, 1].imshow(tgt_avg_alt_mat,cmap='plasma')
        axes[0, 1].set_title('Average Altitude (Target)')
        axes[0, 1].set_xlabel(f'{n_cols} columns')
        axes[0, 1].set_ylabel(f'{n_rows} rows')

    # Subplot 3: Delta Altitude (Volume Change)
        im3 = axes[1, 0].imshow(delta_alt_mat,cmap='coolwarm')
        axes[1, 0].set_title('Delta Altitude (Volume Change)')
        axes[1, 0].set_xlabel(f'{n_cols} columns')
        axes[1, 0].set_ylabel(f'{n_rows} rows')

    # Colorbars
        fig.colorbar(im1, ax=axes[0, 0])
        fig.colorbar(im2, ax=axes[0, 1])
        fig.colorbar(im3, ax=axes[1, 0])

        plt.tight_layout()
        plt.show()
        
        #start_time = time.time()
        #total_volume_change = 123.45 
        #elapsed_time = time.time() - start_time
        result_label.config(text=f"Total Volume Change: {total_volume_change:.2f} m³\nElapsed Time: {elapsed_time:.2f} seconds")
        return total_volume_change
    
    #root.grid_rowconfigure(0, weight=1)
    #root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    root.mainloop()

def visualize_a_point_cloud(pcd, title):
    try:
        o3d.visualization.draw_geometries([pcd], window_name=title)
    except:
        messagebox.showwarning("Error", "กรุณาเลือกไฟล์ LAS ก่อน")

def start_viewer(pcd1, pcd2, step):
    if pcd1 is not None and pcd2 is not None:
        matrix1 = np.eye(4)
        matrix2 = np.eye(4)
        
        translation_offset = np.array([-0.1,0.0,0.0])  
        pcd1.translate(translation_offset)
        
        draw_interactive(pcd1, pcd2, matrix1, matrix2, step)
    else:
        messagebox.showerror("Error", "Please load both files before starting the viewer.")

def show_man():
    data_man = {
        'key_shortcut': ['A', 'D', 'W', 'S', 'X', 'Y', 'Z', 'Q', 'E'],
        'description': ['Move point cloud to the left', 
                        'Move point cloud to the right', 
                        'Move point cloud forward', 
                        'Move point cloud backward', 
                        'Rotate the point cloud along the X axis.', 
                        'Rotate the point cloud along the Y axis.', 
                        'Rotate the point cloud along the Z axis.', 
                        'Move pointcloud upper',
                        'Move pointcloud lower']
    }
    
    table_window = tk.Toplevel()
    table_window.geometry("400x250")
    table_window.title('Key button')
    table_window.configure(bg='#E1F5FE')
    
    columns = ('Key Shortcut', 'Description')
    tree = ttk.Treeview(table_window, columns=columns, show='headings')
    
    tree.heading('Key Shortcut', text='Key Shortcut') 
    tree.column('Key Shortcut', width=80, anchor='center')
        
    tree.heading('Description', text='Description')
    tree.column('Description', width=200, anchor='w')
    
    keys = data_man['key_shortcut']
    descriptions = data_man['description']
    
    for i in range(len(keys)):
        tree.insert('', 'end', values=(keys[i], descriptions[i]))
    
    tree.pack(expand=True, fill='both', padx=10, pady=10)




def login_window():
    login_win = tk.Toplevel()
    login_win.title("Login")
    login_win.geometry("500x300")
    login_win.configure(bg='#E1F5FE')
    login_label = tk.Label(login_win, text="Login", font=("Helvetica", 16, 'bold'), bg='#E1F5FE', fg="#0277BD")
    login_label.pack(pady=20)

    username_label = tk.Label(login_win, text="Username", font=("Helvetica", 12), bg='#E1F5FE', fg="#0277BD")
    username_label.pack(pady=5)
    username_entry = tk.Entry(login_win, font=("Helvetica", 12))
    username_entry.pack(pady=5)

    password_label = tk.Label(login_win, text="Password", font=("Helvetica", 12), bg='#E1F5FE', fg="#0277BD")
    password_label.pack(pady=5)
    password_entry = tk.Entry(login_win, font=("Helvetica", 12), show="*")
    password_entry.pack(pady=5)

    def on_login():
        username = username_entry.get()
        password = password_entry.get()

        if username == "admin" and password == "1234":
            messagebox.showinfo("Success", "Login Successful!")
            login_win.destroy()
            create_gui()  # Open the main GUI window
        else:
            messagebox.showerror("Error", "Invalid credentials. Please try again.")
    
    login_btn = tk.Button(login_win, text="Login", command=on_login, width=20, height=2)
    login_btn.pack(pady=20)

    login_win.mainloop()

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
    login_window() 
