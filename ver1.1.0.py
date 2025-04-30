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
#import matplotlib.colors as mcolors


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
    root.iconbitmap('/Users/suchayapanchuay/Documents/Project/blankspace/image/logo_pro.png')
    root.title("SeaDan")
    root.geometry("1000x1000")
    
    bg_image = Image.open('/Users/suchayapanchuay/Documents/Project/blankspace/image/Background.jpg')
    bg_image = bg_image.resize((1000, 1000))  # ปรับขนาดรูปภาพให้พอดีกับหน้าต่าง
    bg_photo = ImageTk.PhotoImage(bg_image)
    
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
 
    image = Image.open('/Users/suchayapanchuay/Documents/Project/blankspace/image/Main_Logo.png')  
    image = image.resize((100, 100))  # ปรับขนาดรูปภาพ
    photo = ImageTk.PhotoImage(image)
    
    header_frame = tk.Frame(root, bg="#FFF9F0")
    header_frame.place(relx=0.5, rely=0.07, anchor="center", width=1000, height=185) 
    
    image_label = tk.Label(root, image=photo,bg='#FFF9F0') #bg='#ADE1FB'
    image_label.image = photo  # เก็บอ้างอิงรูปภาพเพื่อไม่ให้ถูกลบ
    image_label.grid(row=0, column=0,columnspan=2, pady=(7, 0), padx=20, sticky="n")
      
    header_label = tk.Label(root, text="Welcome to SeaDan", font=("Helvetica", 21, 'bold'),bg='#FFF9F0',fg="#0F2573") 
    header_label.grid(row=1, column=0, columnspan=2, pady=5, padx=20, sticky="n")
    
    pcd1 = None
    pcd2 = None
    step_value = tk.DoubleVar(value=0.05)
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
                messagebox.showinfo("Success", "File 1 loaded successfully!")
            except Exception as e:
                load_btn1.config(fg="red")  
                messagebox.showerror("Error", f"Failed to load file: {e}")

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
                messagebox.showinfo("Success", "File 2 loaded successfully!")
            except Exception as e:
                load_btn2.config(fg="red",font=("bold")) 
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    load_btn1 = tk.Button(root, text="Load Time Series1", command=load_file1, width=20, height=2)
    load_btn1.grid(row=2, column=0 ,columnspan=2, pady=10, padx=20, sticky="n")

    load_btn2 = tk.Button(root, text="Load Time Series2", command=load_file2, width=20, height=2)
    load_btn2.grid(row=3, column=0 ,columnspan=2, pady=10, padx=20, sticky="n")
    
    step_label = tk.Label(root, text="Step value", font=("Helvetica", 14,"bold"), bg="#F0C38E", fg="#0F2573")
    step_label.grid(row=4,pady=10, padx=5)
    
    transfrom_label = tk.Label(root, text="Transformation Metric", font=("Helvetica", 16,"bold"), bg='#FFF9F0', fg="#0F2573")
    transfrom_label.grid(row=5, column=0, columnspan=2, pady=0, padx=20)

    step_entry = tk.Entry(root, textvariable=step_value)
    step_entry.grid(row=5 , pady=10, padx=5)

    start_vis_source = tk.Button(root, text="Start Visualization Time Series1", command=lambda: visualize_a_point_cloud(pcd1, "source"),width=22, height=2)
    start_vis_target = tk.Button(root, text="Start Visualization Time Series2", command=lambda: visualize_a_point_cloud(pcd2, "target"),width=22, height=2)
    start_vis_source.grid(row=6,pady=10, padx=5)
    start_vis_target.grid(row=7,pady=10, padx=5)

    start_viewer_btn = tk.Button(root, text="Start Visualize & Align", command=lambda: start_viewer(pcd1, pcd2, step_value.get()),width=22, height=2)
    start_viewer_btn.grid(row=8,pady=10, padx=5)

    start_calculate_btn = tk.Button(root, text="Start Calculate Volume", command=lambda:[
            stop_animation.clear(),
            threading.Thread(target=animate_gif, args=(canvas, dolphin_sequence, dolphin_id, stop_animation)).start(),
            threading.Thread(target=start_calculate, args=(pcd1, pcd2)).start(),
            start_progress(stop_animation)
            #threading.Thread(target=start_progress, args=(stop_animation)).start(),
        ],width=22, height=2)
    start_calculate_btn.grid(row=9,pady=10, padx=5)
    
    result_label = tk.Label(root, text="Results will appear here.", font=("Helvetica", 14,"bold"), bg='#FFF9F0', fg="#187C19")
    result_label.grid(row=10, column=0, columnspan=2, pady=0, padx=20)

    man_btn = tk.Button(root, text="Help", command=show_man,width=8, height=2)
    #man_btn.grid(row=10, rowspan=3,pady=10, padx=5)
    man_btn.grid(row=10,columnspan=2,rowspan=3, pady=80, padx=5, sticky="n")
    
    text_frame = tk.Label(root, text="Time Series1\n(ข้อมูลก่อนการเปลี่ยนแปลง)\n\nTime Series2\n(ข้อมูลหลังการเปลี่ยนแปลง)\n\nTime Series1 - Time Series2 = Volumn Change", font=("Helvetica", 12,"bold"), bg="#F0C38E", fg="#0F2573")
    text_frame.grid(row=2, column=1, rowspan=5, padx=(10,5), pady=10, sticky="n")
    text_frame = tk.Label(root, text="Step Value Normal = 0.05 \n(ใช้เมื่อไฟล์มีขนาด มากกว่าKB แต่น้อยกว่า10MB)", font=("Helvetica", 12,"bold"), bg="#F0C38E", fg="#0F2573")
    text_frame.grid(row=5, column=1, rowspan=5, padx=(10,5), pady=10, sticky="n")
    text_frame = tk.Label(root, text="Start Visualization Source \n(ดูเฉพาะไฟล์ Source)\n(ถ้าจะออกจากหน้านี้กด esc)", font=("Helvetica", 12,"bold"), bg="#F0C38E", fg="#0F2573")
    text_frame.grid(row=6, column=1, rowspan=5, padx=(10,5), pady=10, sticky="n")
    text_frame = tk.Label(root, text="Start Visualization Target \n(ดูเฉพาะไฟล์ Target)\n(ถ้าจะออกจากหน้านี้กด esc)", font=("Helvetica", 12,"bold"), bg="#F0C38E", fg="#0F2573")
    text_frame.grid(row=7, column=1, rowspan=5, padx=(10,5), pady=10, sticky="n")
    text_frame = tk.Label(root, text="Start Visualize & Align \n(ดูไฟล์ Source และ Target)", font=("Helvetica", 12,"bold"), bg="#F0C38E", fg="#0F2573")
    text_frame.grid(row=8, column=1, rowspan=5, padx=(10,5), pady=10, sticky="n")
    text_frame = tk.Label(root, text="Start Calculate Volume \n (กราฟคำนวณปริมาตรการเปลี่ยนแปลง)", font=("Helvetica", 12,"bold"), bg="#F0C38E", fg="#0F2573")
    text_frame.grid(row=9, column=1, rowspan=5, padx=(10,5), pady=10, sticky="n")
    
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
        
        
    def start_calculate(src_pcd, target_pcd):
        src_points = np.asarray(src_pcd.points)
        tgt_points = np.asarray(target_pcd.points)

        src_x_min, src_x_max = min(src_points[:, 0]), max(src_points[:, 0])
        src_y_min, src_y_max = min(src_points[:, 1]), max(src_points[:, 1])
        tgt_x_min, tgt_x_max = min(tgt_points[:, 0]), max(tgt_points[:, 0])
        tgt_y_min, tgt_y_max = min(tgt_points[:, 1]), max(tgt_points[:, 1])

        gbl_x_min, gbl_x_max = min(src_x_min, tgt_x_min), max(src_x_max, tgt_x_max)
        gbl_y_min, gbl_y_max = min(src_y_min, tgt_y_min), max(src_y_max, tgt_y_max)

    # Calculate the total width and height in meters
        width_meters = gbl_x_max - gbl_x_min
        height_meters = gbl_y_max - gbl_y_min

        col_width = 1.0  
        row_width = 1.0  
        
        #col_width = 0.1  
        #row_width = 0.1 

        n_cols = math.ceil((gbl_x_max - gbl_x_min) / col_width)
        n_rows = math.ceil((gbl_y_max - gbl_y_min) / row_width)

        start_time = time.perf_counter()

        src_args = (src_points, gbl_x_min, gbl_y_min, col_width, row_width, n_rows, n_cols)
        tgt_args = (tgt_points, gbl_x_min, gbl_y_min, col_width, row_width, n_rows, n_cols)
    
        src_z_min, src_z_max = np.min(src_points[:, 2]), np.max(src_points[:, 2])
        tgt_z_min, tgt_z_max = np.min(tgt_points[:, 2]), np.max(tgt_points[:, 2])

    # คำนวณ Global Altitude Range
        gbl_z_min = min(src_z_min, tgt_z_min)  # ความสูงต่ำสุด
        gbl_z_max = max(src_z_max, tgt_z_max)  # ความสูงสูงสุด

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

    # Delta altitude matrix and volume change calculation
        delta_alt_mat = tgt_avg_alt_mat - src_avg_alt_mat
        cell_area = col_width * row_width

        volume_change = delta_alt_mat * cell_area
        volume_change = np.nan_to_num(volume_change)
        total_volume_change = np.sum(volume_change)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
            
        # Transformation Metrics (using ICP)
        reg_icp = o3d.pipelines.registration.registration_icp(
            src_pcd, target_pcd, 0.005, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        transformation_matrix = reg_icp.transformation
        
        stop_animation.set()  
        
        sand_increase = np.sum(delta_alt_mat[delta_alt_mat > 0.1] * cell_area)
        sand_decrease = np.sum(delta_alt_mat[delta_alt_mat < -0.1] * cell_area)
        
        q.put((src_avg_alt_mat, tgt_avg_alt_mat, delta_alt_mat, gbl_z_min, gbl_z_max, width_meters, height_meters, sand_increase, sand_decrease, count_point_src, count_point_tgt, total_volume_change, elapsed_time,transformation_matrix))
                
        def reset_gui():
            result_text.config(state=tk.NORMAL)
            result_text.delete(1.0, tk.END)
            result_text.config(state=tk.DISABLED)
        
            load_btn1.config(fg="#0F2573", text="Load Time Series1")
            load_btn2.config(fg="#0F2573", text="Load Time Series2")
        
            global src_file, tgt_file
            src_file = None
            tgt_file = None
        
        reset_btn = tk.Button(
            root, 
            text="Reset", 
            command=reset_gui, 
            width=6, 
            height=2, 
            font=("Helvetica", 12, "bold"), 
            bg="#FF6F61", 
            fg="#187C19"
        )
        reset_btn.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)
        graph = tk.Button(root, text="Graph", command=lambda:plot_graph(src_avg_alt_mat, tgt_avg_alt_mat, delta_alt_mat, gbl_z_min, gbl_z_max, width_meters, height_meters, sand_increase, sand_decrease), width=10, height=2)
        graph.grid(row=10, pady=10, padx=5)
        
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
                graph = tk.Button(root, text="Graph", command=lambda:plot_graph(src_avg_alt_mat, tgt_avg_alt_mat, delta_alt_mat, gbl_z_min, gbl_z_max, width_meters, height_meters, sand_increase, sand_decrease), width=10, height=2)
                graph.grid(row=10, pady=10, padx=5)
        
        result_text = tk.Text(root, height=10, width=20, wrap="word", font=("Arial", 17,"bold"))
        result_text.grid(row=6, column=0, columnspan=2, rowspan=5, pady=20, padx=5, sticky="n")

        result_text.config(state=tk.NORMAL)
        result_text.insert(tk.END, f"\n{np.array2string(transformation_matrix, formatter={'float_kind': lambda x: f'{x:.2f}'})}\n\n\n")
        result_text.config(state=tk.DISABLED)
        copy_label = tk.Button(root, text="Copy", command=copy_to_clipboard, fg="blue", cursor="hand2", width=5, height=2,font=("Arial", 13,"bold"))
        copy_label.grid(row=8, column=0, columnspan=2, rowspan=5, pady=20, padx=5, sticky="n")
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
        history_btn = tk.Button(root, text="View More and History", command=show_history, width=17, height=2, font=("Helvetica", 12))
        history_btn.grid(row=10, rowspan=3,pady=10, padx=5)
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

    # Subplot 2: Average Altitude (Target)
        im2 = axes[0, 1].imshow(tgt_avg_alt_mat, cmap='viridis_r', norm=norm)
        axes[0, 1].set_title('Average Altitude (Time Series2)')
        axes[0, 1].set_xlabel(f'Width {width_meters:.2f} meters')
        axes[0, 1].set_ylabel(f'Height {height_meters:.2f} meters')

    # Subplot 3: Delta Altitude (Volume Change)
        im3 = axes[1, 0].imshow(delta_alt_mat, cmap='viridis_r', norm=norm)
        axes[1, 0].set_title('Delta Altitude (Volume Change)')
        axes[1, 0].set_xlabel(f'Width {width_meters:.2f} meters')
        axes[1, 0].set_ylabel(f'Height {height_meters:.2f} meters')

    # Subplot 4: Period of change
        threshold_min = -0.1
        threshold_max = 0.1
        norm1 = Normalize(vmin=threshold_min, vmax=threshold_max)
        im4 = axes[1, 1].imshow(delta_alt_mat, cmap='bwr', norm=norm1)
        axes[1, 1].set_title('Period of change')
        axes[1, 1].set_xlabel('Width (meters)')
        axes[1, 1].set_ylabel('Height (meters)')
        axes[1, 1].text(0.5, -0.15, f'> 0.1 m : Sand Increase(Red)\n< -0.1 m : Sand Decrease(Blue)\n Sand Volume Increase: {sand_increase:.2f} m³\nSand Volume Decrease: {sand_decrease:.2f} m³',
            fontsize=12, ha='center', va='top', transform=axes[1, 1].transAxes)

    # Add colorbars for each graph
        fig.colorbar(im1, ax=axes[0, 0])
        fig.colorbar(im2, ax=axes[0, 1])
        fig.colorbar(im3, ax=axes[1, 0])
        fig.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()
        
    q = queue.Queue()
    
    def start_progress(stop_animation):
            total_tasks = 100
            for i in range(1,total_tasks + 1):
                if stop_animation.is_set():
                        percentage_label.config(text="Complete!")  # เปลี่ยนข้อความเป็น Completed
                        root.update_idletasks() 
                        return
                time.sleep(0.05)
                percentage_label.config(text=f"{i}%")
                percentage_label.update()
            else:
                percentage_label.config(text="Completed!")
                root.update_idletasks()
        
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    
    canvas = tk.Canvas(root, width=50, height=70, bg='#F0C38E')
    
    dolphin_gif = Image.open('/Users/suchayapanchuay/Documents/Project/blankspace/image/gif2.gif')  # ใส่พาธ GIF ของคุณ
    dolphin_sequence = [
        ImageTk.PhotoImage(img.resize((50, 70), Image.Resampling.LANCZOS))
        for img in ImageSequence.Iterator(dolphin_gif)
    ]
    dolphin_id = canvas.create_image(25, 35, anchor="center", image=dolphin_sequence[0])
    canvas.grid(row=10,column=1, rowspan=3, padx=(10,5), pady=10, sticky="n")
    
    #progress_bar = ttk.Progressbar(root, orient="horizontal", length=120, mode="determinate")
    #progress_bar.grid(row=10,column=1, rowspan=3, padx=(10,5), pady=90, sticky="n")
    
    percentage_label = tk.Label(root, text="0%")
    percentage_label.grid(row=10,column=1, rowspan=3, padx=(10,5), pady=90, sticky="n")
    #percentage_label.grid(row=10,column=1, rowspan=3, padx=(190,5), pady=90, sticky="n")
    
    #completion_label = tk.Label(root, text="")
    #completion_label.grid(row=3,pady=10, padx=5)
    
    stop_animation = threading.Event() 

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
    create_gui()
