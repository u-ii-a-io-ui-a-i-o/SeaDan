import laspy
import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from calculate_volume import calculate_grid_median_with_kdtree
import multiprocessing 
import time
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image, ImageTk,ImageSequence
import threading
import queue
from matplotlib import cm
from scipy.spatial import cKDTree
import copy
import threading
import subprocess
#import matplotlib.colors as mcolors
#new


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
    #root.iconbitmap('/Users/suchayapanchuay/Documents/Project/blankspace/image/logo_pro.png')
    root.title("SeaDan")
    root.geometry("2000x2000")
    
    try:
        bg_image = Image.open('/Users/suchayapanchuay/Documents/Project/blankspace/imageฝBackground.jpg')
    except:
        bg_image = Image.open('/Users/suchayapanchuay/Documents/Project/blankspace/image/Background.jpg')
    bg_image = bg_image.resize((2000, 2000))  # ปรับขนาดรูปภาพให้พอดีกับหน้าต่าง
    bg_photo = ImageTk.PhotoImage(bg_image)
    
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
 
    try:
        image = Image.open('/Users/suchayapanchuay/Documents/Project/blankspace/image/Main_Logo.png')  
    except:
        image = Image.open('/Users/suchayapanchuay/Documents/Project/blankspace/image/Main_Logo.png')
    image = image.resize((100, 100))  # ปรับขนาดรูปภาพ
    photo = ImageTk.PhotoImage(image)
    
    header_frame = tk.Frame(root, bg="#FFF9F0")
    header_frame.place(relx=0.5, rely=0.07, anchor="center", width=2000, height=185) 
    
    image_label = tk.Label(root, image=photo,bg='#FFF9F0') #bg='#ADE1FB'
    image_label.image = photo  # เก็บอ้างอิงรูปภาพเพื่อไม่ให้ถูกลบ
    image_label.grid(row=0, column=0,columnspan=2, pady=(7, 0), padx=20, sticky="n")
      
    header_label = tk.Label(root, text="SeaDan", font=("Helvetica", 21, 'bold'),bg='#FFF9F0',fg="#0F2573") 
    header_label.grid(row=1, column=0, columnspan=2, pady=3, padx=20, sticky="n")
    
    pcd1 = None
    pcd2 = None
    step_value = tk.DoubleVar(value=0.05)
    grid_value = tk.DoubleVar(value=0.05)
    last_loaded_file1 = None  
    last_loaded_file2 = None
    
    def load_file1():
        nonlocal pcd1,last_loaded_file1
        file_path = pickfile("source")
        if file_path:
            if file_path == last_loaded_file1:  
                messagebox.showinfo("Info", "File 1 is already loaded.")
                return

            try:
                las_data, pcd = read_las_file(file_path)
                points, colors = las2Array(las_data)
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd1 = pcd
                last_loaded_file1 = file_path  
                load_btn1.config(fg="green")  
                file_label1.config(text=f"✅ Loaded: {file_path.split('/')[-1]}")
                messagebox.showinfo("Success", "File 1 loaded successfully!")
                
                progress_bar.grid_forget()
            except Exception as e:
                load_btn1.config(fg="red")  
                messagebox.showerror("Error", f"Failed to load file: {e}")
                progress_bar.grid_forget()

    def load_file2():
        nonlocal pcd2,last_loaded_file2
        file_path = pickfile("target")
        if file_path:
            if file_path == last_loaded_file2:  
                messagebox.showinfo("Info", "File 2 is already loaded.")
                return

            try:

                las_data, pcd = read_las_file(file_path)
                points, colors = las2Array(las_data)
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd2 = pcd
                last_loaded_file2 = file_path  
                load_btn2.config(fg="green",font=("bold"))
                file_label2.config(text=f"✅ Loaded: {file_path.split('/')[-1]}")  
                messagebox.showinfo("Success", "File 2 loaded successfully!")
                progress_bar.grid_forget()
            except Exception as e:
                load_btn2.config(fg="red",font=("bold")) 
                messagebox.showerror("Error", f"Failed to load file: {e}")
                progress_bar.grid_forget()
    
    load_btn1 = tk.Button(root, text="📂 Load Time Series1", command=load_file1, width=20, height=2,background="lightblue")
    load_btn1.grid(row=2, column=0 , pady=10, padx=10)

    load_btn2 = tk.Button(root, text="📂 Load Time Series2", command=load_file2, width=20, height=2,background="lightblue")
    load_btn2.grid(row=2, column=1, pady=10, padx=10)
    
    progress_bar = ttk.Progressbar(root, length=300, mode="determinate")
    
    file_label1 = tk.Label(root, text="❌ No file loaded", fg="blue",bg="white")
    file_label1.grid(row=3, column=0, pady=5)
    
    file_label2 = tk.Label(root, text="❌ No file loaded", fg="blue",bg="white")
    file_label2.grid(row=3, column=1, pady=5)
    
    #step_label = tk.Label(root, text="Step value", font=("Helvetica", 14,"bold"), bg="#F0C38E", fg="#0F2573")
    #step_label.grid(row=3,column=0, columnspan=2, pady=10)
    #
    #step_entry = tk.Entry(root, textvariable=step_value)
    #step_entry.grid(row=4,column=0, columnspan=2, pady=10)
    
    step_label = tk.Label(root, text="Step value", font=("Helvetica", 14,"bold"), bg="#F0C38E", fg="#0F2573")
    step_label.grid(row=3,column=0, columnspan=2, pady=10,sticky="w",padx=570)
    
    step_entry = tk.Entry(root, textvariable=step_value)
    step_entry.grid(row=4,column=0, columnspan=2, pady=10,sticky="w",padx=510)
    
    grid_size = tk.Label(root, text="Grid size", font=("Helvetica", 14,"bold"), bg="#F0C38E", fg="#0F2573")
    grid_size.grid(row=3,column=0, columnspan=2, pady=10,sticky="e",padx=570)
    
    grid_entry = tk.Entry(root, textvariable=grid_value)
    grid_entry.grid(row=4,column=0, columnspan=2, pady=10,sticky="e",padx=510)
    
    transfrom_label = tk.Label(root, text="Result Transformation Metric", font=("Helvetica", 16,"bold"), bg='#FFF9F0', fg="#0F2573")
    transfrom_label.grid(row=8, column=0, columnspan=2, pady=0, padx=20)

    start_vis_source = tk.Button(root, text="Start Visualization Time Series1", command=lambda: visualize_a_point_cloud(pcd1, "Time Series1"),width=22, height=2)
    start_vis_target = tk.Button(root, text="Start Visualization Time Series2", command=lambda: visualize_a_point_cloud(pcd2, "Time Series2"),width=22, height=2)
    start_vis_source.grid(row=6,pady=10, padx=5)
    start_vis_target.grid(row=6,column=0, columnspan=2, pady=10)

    start_viewer_btn = tk.Button(root, text="Start Visualize & Align", command=lambda: start_viewer(pcd1, pcd2, step_value.get()),width=22, height=2)
    start_viewer_btn.grid(row=6, column=1, pady=10, padx=10)

    progress_queue = queue.Queue()
    start_calculate_btn = tk.Button(root, text="🚀 Start Calculate Volume", command=lambda:[
            stop_animation.clear(),
            threading.Thread(target=animate_gif, args=(canvas, dolphin_sequence, dolphin_id, stop_animation)).start(),
            threading.Thread(target=start_calculate, args=(pcd1, pcd2, progress_queue, grid_value.get())).start(),
            start_progress(stop_animation,progress_queue)
            #threading.Thread(target=start_progress, args=(stop_animation)).start(),
        ],width=22, height=2)
    start_calculate_btn.grid(row=2, column=0, columnspan=2, pady=10)
    
    #result_label = tk.Label(root, text="Results will appear here.", font=("Helvetica", 14,"bold"), bg='#FFF9F0', fg="#187C19")
    #result_label.grid(row=12, column=0, columnspan=2, pady=0, padx=20)

    man_btn = tk.Button(root, text="⚙️ User manual", command=show_man,width=18, height=2)
    #man_btn.grid(row=10, rowspan=3,pady=10, padx=5)
    man_btn.grid(row=14,pady=10,padx=5)
    squear = tk.Canvas(root, width=200, height=200, bg="black")
    squear.grid(row=9, column=0, columnspan=2, rowspan=5, pady=20, padx=5, sticky="n")
    
    text_frame = tk.Label(root, text="Time Series1\n(ข้อมูลก่อนการเปลี่ยนแปลง)\n\nTime Series2\n(ข้อมูลหลังการเปลี่ยนแปลง)\n\nTime Series1 - Time Series2 = Volumn Change\n\nStart Calculate Volume \n(เริ่มต้นการคำนวณ)\n\nStep Value Normal = 0.05 \n(ใช้เมื่อไฟล์มีขนาด มากกว่าKB แต่น้อยกว่า10MB)\n",
                          font=("Helvetica", 12,"bold"), bg="#F0C38E", fg="#0F2573")
    text_frame.grid(row=9, pady=10, rowspan=5,padx=5)
    text_frame = tk.Label(root, text="    Start Visualization Time Series1    \n(ดูเฉพาะไฟล์ Time Series1)\n\n    Start Visualization Time Series2    \n(ดูเฉพาะไฟล์ Time Series2)\n\n  Start Visualize & Align  \n(ดูไฟล์ Time Series1 และ Time Series2)\n\n**ถ้าจะออกจากหน้านี้กด esc**", font=("Helvetica", 12,"bold"), bg="#F0C38E", fg="#0F2573")
    text_frame.grid(row=9, rowspan=5,column=1, pady=10, padx=10)
    #text_frame = tk.Label(root, text="Start Visualize & Align \n(ดูไฟล์ Source และ Target)", font=("Helvetica", 12,"bold"), bg="#F0C38E", fg="#0F2573")
    #text_frame.grid(row=8, column=1, rowspan=5, padx=(10,5), pady=10, sticky="n")
    
    history_data = []  # ลิสต์สำหรับเก็บประวัติการคำนวณ

    def show_history():
        history_window = tk.Toplevel(root)  # สร้างหน้าต่างใหม่สำหรับประวัติ
        history_window.title("History of Calculations")
        history_window.geometry("600x400")
        
        history_text = tk.Text(history_window, height=100, width=100, wrap="word", font=("Arial", 12))
        history_text.pack(pady=10)

        history_text.insert(tk.END, "Calculation History:\n")
        if len(history_data) == 0:
            history_text.insert(tk.END, "No history data available.")  # ถ้าไม่มีข้อมูลให้แสดงข้อความนี้
        for entry in history_data:
            history_text.insert(tk.END, f"{entry}\n\n")
        history_text.config(state=tk.DISABLED)
    
    def animate_gif(canvas, gif_sequence, gif_id, stop_flag):
        frame_index = 0

        def next_frame():
            nonlocal frame_index
            if stop_flag.is_set():  # ถ้าถูกตั้งค่าให้หยุด ออกจากแอนิเมชัน
                return
            frame_index = (frame_index + 1) % len(gif_sequence)
            canvas.itemconfig(gif_id, image=gif_sequence[frame_index])
            canvas.after(100, next_frame)

        next_frame()
        
        
    def start_calculate(src_pcd, target_pcd,progress_queue, grid_size=1.0):
        total_steps = 10
        current_step = 0
        src_points = np.asarray(src_pcd.points)
        tgt_points = np.asarray(target_pcd.points)
        
        def update_progress():
            nonlocal current_step
            current_step += 1
            percentage = int((current_step / total_steps) * 100)
            progress_queue.put(percentage)
            
        update_progress()

        src_x_min, src_x_max = min(src_points[:, 0]), max(src_points[:, 0])
        src_y_min, src_y_max = min(src_points[:, 1]), max(src_points[:, 1])
        tgt_x_min, tgt_x_max = min(tgt_points[:, 0]), max(tgt_points[:, 0])
        tgt_y_min, tgt_y_max = min(tgt_points[:, 1]), max(tgt_points[:, 1])
        
        update_progress()

        gbl_x_min, gbl_x_max = min(src_x_min, tgt_x_min), max(src_x_max, tgt_x_max)
        gbl_y_min, gbl_y_max = min(src_y_min, tgt_y_min), max(src_y_max, tgt_y_max)
        
        update_progress()

    # Calculate the total width and height in meters
        width_meters = gbl_x_max - gbl_x_min
        height_meters = gbl_y_max - gbl_y_min

        col_width = grid_size
        row_width = grid_size
        
        #col_width = 0.1  
        #row_width = 0.1 

        n_cols = math.ceil((gbl_x_max - gbl_x_min) / col_width)
        n_rows = math.ceil((gbl_y_max - gbl_y_min) / row_width)

        start_time = time.perf_counter()
        
        update_progress()

        src_args = (src_points, gbl_x_min, gbl_y_min, col_width, row_width, n_rows, n_cols)
        tgt_args = (tgt_points, gbl_x_min, gbl_y_min, col_width, row_width, n_rows, n_cols)
    
        src_z_min, src_z_max = np.min(src_points[:, 2]), np.max(src_points[:, 2])
        tgt_z_min, tgt_z_max = np.min(tgt_points[:, 2]), np.max(tgt_points[:, 2])

    # คำนวณ Global Altitude Range
        gbl_z_min = min(src_z_min, tgt_z_min)  # ความสูงต่ำสุด
        gbl_z_max = max(src_z_max, tgt_z_max)  # ความสูงสูงสุด
        
        update_progress()

        if n_cols * n_rows < 1:
        # Use multiprocessing to calculate median altitudes for both source and target
            with multiprocessing.Pool(processes=4) as pool:
                results = pool.starmap(calculate_grid_median_with_kdtree, [src_args, tgt_args])
                pool.close() 
                pool.join()
        # Unpack results
            src_avg_alt_mat, count_point_src = results[0]
            tgt_avg_alt_mat, count_point_tgt = results[1]
        else:
            src_avg_alt_mat, count_point_src = calculate_grid_median_with_kdtree(*src_args)
            tgt_avg_alt_mat, count_point_tgt = calculate_grid_median_with_kdtree(*tgt_args)

        print(count_point_src)
        print(count_point_tgt)
        
        update_progress()

    # Delta altitude matrix and volume change calculation
        delta_alt_mat = tgt_avg_alt_mat - src_avg_alt_mat
        cell_area = col_width * row_width
        
        update_progress()

        volume_change = delta_alt_mat * cell_area
        volume_change = np.nan_to_num(volume_change)
        total_volume_change = np.sum(volume_change)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        update_progress()
            
        # Transformation Metrics (using ICP)
        reg_icp = o3d.pipelines.registration.registration_icp(
            # src_pcd, target_pcd, 0.005, np.eye(4),
            target_pcd, src_pcd, 0.005, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        transformation_matrix = reg_icp.transformation
        
        update_progress()
        
        stop_animation.set()  
        
        sand_increase = np.sum(delta_alt_mat[delta_alt_mat > 0.1] * cell_area)
        sand_decrease = np.sum(delta_alt_mat[delta_alt_mat < -0.1] * cell_area)
        
        q = queue.Queue()
        
        q.put((src_avg_alt_mat, tgt_avg_alt_mat, delta_alt_mat, gbl_z_min, gbl_z_max, width_meters, height_meters, sand_increase, sand_decrease, count_point_src, count_point_tgt, total_volume_change, elapsed_time,transformation_matrix,step_value.get(),col_width, row_width))
                
        def reset_gui():
            result_text.config(state=tk.NORMAL)
            result_text.delete(1.0, tk.END)
            result_text.config(state=tk.DISABLED)
        
            load_btn1.config(fg="#0F2573", text="📂 Load Time Series1")
            load_btn2.config(fg="#0F2573", text="📂 Load Time Series2")
        
            global src_file, tgt_file
            src_file = None
            tgt_file = None
        
        reset_btn = tk.Button(
            root, 
            text="Reset", 
            command=reset_gui, 
            width=6, 
            height=2, 
            font=("Helvetica", 13, "bold"), 
            bg="#FF6F61", 
            fg="#187C19"
        )
        reset_btn.grid(row=14,column=0, columnspan=2, rowspan=5, pady=5, padx=5, sticky="n")
        graph = tk.Button(root, text="📊 Graph", command=lambda:plot_graph(src_avg_alt_mat, tgt_avg_alt_mat, delta_alt_mat, gbl_z_min, gbl_z_max, width_meters, height_meters, sand_increase, sand_decrease), width=22, height=2)
        graph.grid(row=4, pady=10, padx=5,rowspan=2)
          
        view = tk.Button(root, text="Volume Change", command=lambda: view_all(pcd1,pcd2), width=22, height=2)
        view.grid(row=7,column=1, pady=10, padx=10)
        
        ts1 = tk.Button(root, text="Time Series1",command=lambda:visualize_a_point_cloud_color(pcd1, "Time Series1",apply_cmap=True),width=22, height=2)
        ts1.grid(row=7,pady=10, padx=5)
        
        ts2 = tk.Button(root, text="Time Series2",command=lambda:visualize_a_point_cloud_color(pcd2, "Time Series2",apply_cmap=True),width=22, height=2)
        ts2.grid(row=7,column=0, columnspan=2, pady=10)
        
        def view_all(pcd1, pcd2, threshold_min=-0.1, threshold_max=0.1):
                try:
                    temp_file1 = "pcd1_volume.ply"
                    temp_file2 = "pcd2_volume.ply"

                    o3d.io.write_point_cloud(temp_file1, pcd1)
                    o3d.io.write_point_cloud(temp_file2, pcd2)

                    subprocess.Popen(["python", "visualize_volume.py", temp_file1, temp_file2, str(threshold_min), str(threshold_max)], start_new_session=True)
                except:
                    messagebox.showwarning("Error", "กรุณาเลือกไฟล์ LAS ก่อน")
        
        #def apply_colormap(pcd, colormap_name='viridis_r'):
        #    points = np.asarray(pcd.points)
        #    z_values = points[:, 2]  
        #    colormap = plt.get_cmap(colormap_name)
        #    normalized_z = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
        #    colors = colormap(normalized_z)[:, :3] 
        #    pcd.colors = o3d.utility.Vector3dVector(colors)
                
        def visualize_a_point_cloud_color(pcd, title, apply_cmap=False):
            try:
                # original_colors_pcd1 = np.asarray(pcd1.colors).copy() if pcd1.colors else np.zeros_like(np.asarray(pcd1.points))
                # original_colors_pcd2 = np.asarray(pcd2.colors).copy() if pcd2.colors else np.zeros_like(np.asarray(pcd2.points))
                # if apply_cmap:
                #     apply_colormap(pcd, colormap_name)
                # o3d.visualization.draw_geometries([pcd], window_name=title+"color?")
                
                # pcd1.colors = o3d.utility.Vector3dVector(original_colors_pcd1)
                # pcd2.colors = o3d.utility.Vector3dVector(original_colors_pcd2)
                temp_file = f"{title}_color.ply"
                o3d.io.write_point_cloud(temp_file, pcd)

                subprocess.Popen(["python", "visualize_color.py", temp_file, title, str(apply_cmap)], start_new_session=True)
    
            except:
                messagebox.showwarning("Error", "กรุณาเลือกไฟล์ LAS ก่อน")
        
        def copy_to_clipboard():
            text = result_text.get("1.0", tk.END).strip()  # ดึงข้อความทั้งหมดจาก Text widget
            if text:
                root.clipboard_clear()  # ล้างข้อมูลเก่าบนคลิปบอร์ด
                root.clipboard_append(text)  # เพิ่มข้อความใหม่ในคลิปบอร์ด
                root.update()  # อัปเดต GUI
                messagebox.showinfo("Success", "Text copied to clipboard!")  # แสดงข้อความแจ้งเตือน
            else:
                messagebox.showwarning("Warning", "No text to copy!")
                #   reset_btn.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)
                graph = tk.Button(root, text="Graph", command=lambda:plot_graph(src_avg_alt_mat, tgt_avg_alt_mat, delta_alt_mat, gbl_z_min, gbl_z_max, width_meters, height_meters, sand_increase, sand_decrease), width=22, height=2)
                graph.grid(row=4, pady=10, padx=5,rowspan=2)
        
        result_text = tk.Text(root, height=10, width=30, wrap="word", font=("Arial", 17,"bold"))
        result_text.grid(row=9, column=0, columnspan=2, rowspan=5, pady=20, padx=5, sticky="n")

        result_text.config(state=tk.NORMAL)
        result_text.tag_configure("center", justify="center")
        result_text.insert(tk.END, f"\n{np.array2string(transformation_matrix, formatter={'float_kind': lambda x: f'{x:.2f}'})}\n\n\n")
        result_text.config(state=tk.DISABLED)
        copy_label = tk.Button(root, text="📋", command=copy_to_clipboard, fg="blue", cursor="hand2",font=("Arial", 13,"bold"))
        copy_label.grid(row=12 , column=0, columnspan=2, rowspan=5, pady=25, padx=6, sticky="n")
        copy_label.bind("<Button-1>", copy_to_clipboard)
        #copy_button = tk.Button(root, text="Copy", command=copy_to_clipboard, width=5,height=1 )
        #copy_button.grid(row=9, column=0, columnspan=2, rowspan=5, pady=20, padx=5, sticky="n")
        result_text.bind("<Button-3>", copy_to_clipboard)
        
    
        history_entry = (f"Transformation Metrics\n\n{np.array2string(transformation_matrix, formatter={'float_kind': lambda x: f'{x:.2f}'})}\n\n\n"
                      f"Total Volume Change: {total_volume_change:.2f} m³\nElapsed Time: {elapsed_time:.2f} seconds\n"
                      f"Global Altitude Range: {gbl_z_min:.2f} to {gbl_z_max:.2f}\n"
                      f"----------------------------------------------------------------------------------------------------------------------------------------------------")
        history_data.append(history_entry)
        print("History Entry Added:", history_entry)  # ตรวจสอบข้อมูลที่เพิ่มเข้าไป

        # เพิ่มปุ่ม "View History"
        history_btn = tk.Button(root, text="⏳ View More and History", command=show_history, width=22, height=2)
        history_btn.grid(row=4, column=1, pady=10, padx=10,rowspan=2)
        #history_btn.grid(row=11,pady=10, padx=5)
    
        return total_volume_change, reset_gui

    
    def plot_graph(src_avg_alt_mat, tgt_avg_alt_mat, delta_alt_mat, gbl_z_min, gbl_z_max, width_meters, height_meters, sand_increase, sand_decrease):
        norm = Normalize(vmin=gbl_z_min, vmax=gbl_z_max)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[1, 1].axis('off')
        # Subplot 1: Average Altitude (Source)
        im1 = axes[0, 0].imshow(src_avg_alt_mat, cmap='viridis_r', norm=norm)
        axes[0, 0].set_title('Average Altitude (Time Series1)')
        axes[0, 0].set_xlabel(f'Width {width_meters:.2f} meters')
        axes[0, 0].set_ylabel(f'Height {height_meters:.2f} meters')
        axes[0, 0].invert_yaxis()

    # Subplot 2: Average Altitude (Target)
        im2 = axes[0, 1].imshow(tgt_avg_alt_mat, cmap='viridis_r', norm=norm)
        axes[0, 1].set_title('Average Altitude (Time Series2)')
        axes[0, 1].set_xlabel(f'Width {width_meters:.2f} meters')
        axes[0, 1].set_ylabel(f'Height {height_meters:.2f} meters')
        axes[0, 1].invert_yaxis()

    # Subplot 3: Delta Altitude (Volume Change)
        im3 = axes[1, 0].imshow(delta_alt_mat, cmap='viridis_r', norm=norm)
        axes[1, 0].set_title('Delta Altitude (Volume Change)')
        axes[1, 0].set_xlabel(f'Width {width_meters:.2f} meters')
        axes[1, 0].set_ylabel(f'Height {height_meters:.2f} meters')
        axes[1, 0].invert_yaxis()

    # Subplot 4: Period of change
        threshold_min = -0.1
        threshold_max = 0.1
        norm1 = Normalize(vmin=threshold_min, vmax=threshold_max)
        im4 = axes[1, 1].imshow(delta_alt_mat, cmap='bwr', norm=norm1)
        axes[1, 1].set_title('Period of change')
        axes[1, 1].set_xlabel('Width (meters)')
        axes[1, 1].set_ylabel('Height (meters)')
        axes[1, 1].invert_yaxis()
        axes[1, 1].text(0.5, -0.15, f'> 0.1 m : Sand Increase(Red)\n< -0.1 m : Sand Decrease(Blue)\n Sand Volume Increase: {sand_increase:.2f} m³\nSand Volume Decrease: {sand_decrease:.2f} m³',
            fontsize=12, ha='center', va='top', transform=axes[1, 1].transAxes)

    # Add colorbars for each graph
        fig.colorbar(im1, ax=axes[0, 0])
        fig.colorbar(im2, ax=axes[0, 1])
        fig.colorbar(im3, ax=axes[1, 0])
        fig.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()
    
    def start_progress(stop_animation,progress_queue):
            def update_progress():
                if not progress_queue.empty():
                    percentage = progress_queue.get()
                    percentage_label.config(text=f"{percentage}%")
                    percentage_label.update()
                if not stop_animation.is_set():  
                    root.after(100, update_progress)  # อัปเดตทุก 100ms
                else:
                    percentage_label.config(text="Completed!")  

            update_progress()
    
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    
    canvas = tk.Canvas(root, width=50, height=70, bg='#F0C38E')
    
    try:
        dolphin_gif = Image.open('/Users/suchayapanchuay/Documents/Project/blankspace/image/gif2.gif')  # ใส่พาธ GIF ของคุณ
    except:
        dolphin_gif = Image.open('/Users/suchayapanchuay/Documents/Project/blankspace/image/gif2.gif')  # ใส่พาธ GIF ของคุณ
    dolphin_sequence = [
        ImageTk.PhotoImage(img.resize((50, 70), Image.Resampling.LANCZOS))
        for img in ImageSequence.Iterator(dolphin_gif)
    ]
    dolphin_id = canvas.create_image(25, 35, anchor="center", image=dolphin_sequence[0])
    canvas.place(relx=1.0, rely=1.0, anchor='se', x=-25, y=-30)
    
    percentage_label = tk.Label(root, text="0%")
    percentage_label.place(relx=1.0, rely=1.0, anchor='se', x=-20, y=-10)
    
    stop_animation = threading.Event() 

    root.mainloop()

def visualize_a_point_cloud(pcd, title):
    try:
        # o3d.visualization.draw_geometries([pcd], window_name=title)
        temp_file = f"{title}.ply"
        o3d.io.write_point_cloud(temp_file, pcd)

        subprocess.Popen(["python", "visualize.py", temp_file, title], start_new_session=True)
        
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
    table_window.configure(bg="#F1AA9B")
    
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
# def key_callback_1(vis, pcd1, matrix1, step):
#     vis.register_key_callback(65, lambda vis: move_cloud(vis, pcd1, "left", matrix1, step))     #A
#     vis.register_key_callback(68, lambda vis: move_cloud(vis, pcd1, "right", matrix1, step))    #D
#     vis.register_key_callback(87, lambda vis: move_cloud(vis, pcd1, "up", matrix1, step))       #W
#     vis.register_key_callback(83, lambda vis: move_cloud(vis, pcd1, "down", matrix1, step))     #S
#     vis.register_key_callback(88, lambda vis: rotate_cloud(vis, pcd1, "x"))                     #X
#     vis.register_key_callback(89, lambda vis: rotate_cloud(vis, pcd1, "y"))                     #Y
#     vis.register_key_callback(90, lambda vis: rotate_cloud(vis, pcd1, "z"))                     #Z
#     vis.register_key_callback(88, lambda vis: rotate_cloud(vis, pcd1))                          #X
#     vis.register_key_callback(89, lambda vis: rotate_cloud(vis, pcd1))                          #Y
#     vis.register_key_callback(90, lambda vis: rotate_cloud(vis, pcd1))                          #Z
#     vis.register_key_callback(81, lambda vis: move_cloud(vis, pcd1, "forward", matrix1, step))  #Q
#     vis.register_key_callback(69, lambda vis: move_cloud(vis, pcd1, "backward", matrix1, step)) #E

def key_callback_1(vis, pcd2, matrix2, step):
    vis.register_key_callback(65, lambda vis: move_cloud(vis, pcd2, "left", matrix2, step))     #A
    vis.register_key_callback(68, lambda vis: move_cloud(vis, pcd2, "right", matrix2, step))    #D
    vis.register_key_callback(87, lambda vis: move_cloud(vis, pcd2, "up", matrix2, step))       #W
    vis.register_key_callback(83, lambda vis: move_cloud(vis, pcd2, "down", matrix2, step))     #S
    vis.register_key_callback(88, lambda vis: rotate_cloud(vis, pcd2, "x"))                     #X
    vis.register_key_callback(89, lambda vis: rotate_cloud(vis, pcd2, "y"))                     #Y
    vis.register_key_callback(90, lambda vis: rotate_cloud(vis, pcd2, "z"))                     #Z
    vis.register_key_callback(81, lambda vis: move_cloud(vis, pcd2, "forward", matrix2, step))  #Q
    vis.register_key_callback(69, lambda vis: move_cloud(vis, pcd2, "backward", matrix2, step)) #E

# def key_callback_2(vis, pcd2, matrix2, step):
#     vis.register_key_callback(74, lambda vis: move_cloud(vis, pcd2, "left", matrix2, step))     #J
#     vis.register_key_callback(76, lambda vis: move_cloud(vis, pcd2, "right", matrix2, step))    #L
#     vis.register_key_callback(73, lambda vis: move_cloud(vis, pcd2, "up", matrix2, step))       #I
#     vis.register_key_callback(75, lambda vis: move_cloud(vis, pcd2, "down", matrix2, step))     #K
#     vis.register_key_callback(85, lambda vis: rotate_cloud(vis, pcd2))                          #U
#     vis.register_key_callback(79, lambda vis: move_cloud(vis, pcd2, "forward", matrix2, step))  #O
#     vis.register_key_callback(80, lambda vis: move_cloud(vis, pcd2, "backward", matrix2, step)) #P

def draw(pcd1, pcd2, matrix1, matrix2, step):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)

    # key_callback_1(vis, pcd1, matrix1, step)
    key_callback_1(vis, pcd2, matrix2, step)
    # key_callback_2(vis, pcd2, matrix2, step)

    vis.run()
    vis.destroy_window()
    
def draw_interactive(pcd1, pcd2, matrix1, matrix2, step):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)

    # key_callback_1(vis, pcd1, matrix1, step)
    key_callback_1(vis, pcd2, matrix2, step)
    # key_callback_2(vis, pcd2, matrix2, step)

    vis.run()
    vis.destroy_window()



if __name__ == "__main__":
    create_gui()
