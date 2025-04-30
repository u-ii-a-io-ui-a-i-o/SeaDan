import laspy
import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
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
    root.geometry("400x500")
    root.configure(bg='#F0F0F0')
    
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    style.configure("TLabel", font=("Helvetica", 12), background='#F0F0F0')

    pcd1 = None
    pcd2 = None
    step_value = tk.DoubleVar(value=0.005)

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

    load_btn1 = tk.Button(root, text="Load Source", command=load_file1)
    load_btn1.pack(pady=10)

    load_btn2 = tk.Button(root, text="Load Target", command=load_file2)
    load_btn2.pack(pady=10)

    step_label = tk.Label(root, text="Step value:")
    step_label.pack(pady=5)

    step_entry = tk.Entry(root, textvariable=step_value)
    step_entry.pack(pady=5)
    start_vis_source = tk.Button(root, text="Start Visualization Source", command=lambda: visualize_a_point_cloud(pcd1, "source"))
    start_vis_target = tk.Button(root, text="Start Visualization Target", command=lambda: visualize_a_point_cloud(pcd2, "target"))
    start_vis_source.pack(pady=20)
    start_vis_target.pack(pady=20)

    start_viewer_btn = tk.Button(root, text="Start visualize & align", command=lambda: start_viewer(pcd1, pcd2, step_value.get()))
    start_viewer_btn.pack(pady=20)

    start_calculate_btn = tk.Button(root, text="Start Calculate Volume", command=lambda: start_calculate(pcd1, pcd2))
    start_calculate_btn.pack(pady=20)

    man_btn = tk.Button(root, text="Help", command=show_man)
    man_btn.pack(pady=10)


    root.mainloop()

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

def start_calculate(src_pcd, target_pcd):
    #Extract point coordinates
    src_points = np.asarray(src_pcd.points)
    tgt_points = np.asarray(target_pcd.points)

    #Bounding box and drid setup
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

    if n_cols*n_rows < 1 :
        # Use multiprocessing to calculate median altitudes for both source and target
        with multiprocessing.Pool(processes = 4) as pool:
            # starmap will unpack each tuple argument list correctly
            results = pool.starmap(calculate_grid_median_with_kdtree, [src_args, tgt_args])
        
        # Unpack results
        src_avg_alt_mat, count_point_src = results[0]
        tgt_avg_alt_mat, count_point_tgt = results[1]
        
    else :

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

    # Plotting results
    """
    #fig = plt.figure()
    fig, ax = plt.subplots()
    im = ax.imshow(src_avg_alt_mat)
    ax.set_title('Average Altitude (Source)')
    ax.set_xlabel(f'{n_cols} columns')
    ax.set_ylabel(f'{n_rows} rows')
    plt.show()
    
    fig, ax = plt.subplots()
    im = ax.imshow(tgt_avg_alt_mat)
    ax.set_title('Average Altitude (Target)')
    ax.set_xlabel(f'{n_cols} columns')
    ax.set_ylabel(f'{n_rows} rows')
    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(delta_alt_mat)
    ax.set_title('Delta Altitude (Target - Source)')
    ax.set_xlabel(f'{n_cols} columns')
    ax.set_ylabel(f'{n_rows} rows')
    plt.show()
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ซ่อน subplot ที่ไม่ได้ใช้ (แถว 2 คอลัมน์ 2)
    axes[1, 1].axis('off')

    # Subplot 1: Average Altitude (Source)
    im1 = axes[0, 0].imshow(src_avg_alt_mat)
    axes[0, 0].set_title('Average Altitude (Source)')
    axes[0, 0].set_xlabel(f'{n_cols} columns')
    axes[0, 0].set_ylabel(f'{n_rows} rows')

    # Subplot 2: Average Altitude (Target)
    im2 = axes[0, 1].imshow(tgt_avg_alt_mat)
    axes[0, 1].set_title('Average Altitude (Target)')
    axes[0, 1].set_xlabel(f'{n_cols} columns')
    axes[0, 1].set_ylabel(f'{n_rows} rows')

    # Subplot 3: Delta Altitude (Target - Source)
    im3 = axes[1, 0].imshow(delta_alt_mat)
    axes[1, 0].set_title('Delta Altitude (Target - Source)')
    axes[1, 0].set_xlabel(f'{n_cols} columns')
    axes[1, 0].set_ylabel(f'{n_rows} rows')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the combined plot
    plt.show()
    
    print(f'Total volume change: {total_volume_change:.10f} cubic units')
    print(f"Elapsed time with KD-tree optimization: {elapsed_time:.4f} seconds")

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

'''
def key_callback_1(vis, pcd1, matrix1, step):
    vis.register_key_callback(263, lambda vis: move_cloud(vis, pcd1, "left", matrix1, step))     #left arrow
    vis.register_key_callback(262, lambda vis: move_cloud(vis, pcd1, "right", matrix1, step))    #right arrow
    vis.register_key_callback(265, lambda vis: move_cloud(vis, pcd1, "up", matrix1, step))       #up arrow
    vis.register_key_callback(264, lambda vis: move_cloud(vis, pcd1, "down", matrix1, step))     #down arrow
    vis.register_key_callback(88, lambda vis: rotate_cloud(vis, pcd1, "x"))                          #X
    vis.register_key_callback(89, lambda vis: rotate_cloud(vis, pcd1, "y"))                          #Y
    vis.register_key_callback(90, lambda vis: rotate_cloud(vis, pcd1, "z"))                          #Z
    vis.register_key_callback(81, lambda vis: move_cloud(vis, pcd1, "forward", matrix1, step))  #Q
    vis.register_key_callback(69, lambda vis: move_cloud(vis, pcd1, "backward", matrix1, step)) #E
'''

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
