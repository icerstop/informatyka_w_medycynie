import tkinter as tk
from tkinter import filedialog, Scale, messagebox, Toplevel, Entry, Label, Button, Checkbutton, BooleanVar
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from algorithms import radon_all, inverse_radon_all, calculate_rmse
import pydicom
from dicom_handler import save_as_dicom
import subprocess
import webbrowser
import os
import shlex
import sys

class CTScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CT Scanner Simulation")
        self.loaded_image_path = None
        
        # Parameters
        self.angle_step = 1.0
        self.detector_count = 180
        self.span_angle = 180
        self.use_filter = False
        self.current_angle = 0
        self.is_animation_running = False
        
        
        # Image data
        self.original_image = None
        self.sinogram = None
        self.reconstructed_image = None
        self.current_sinogram = None
        self.current_reconstruction = None
        self.angles = None
        self.animation_id = None
        self.full_sinogram = None
        
        # Set up UI components
        self.setup_ui()
    
    def setup_ui(self):
        # Main frames
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, expand=True, fill=tk.BOTH)
        
        # Control panels
        control_frame = tk.LabelFrame(left_frame, text="Controls", padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        param_frame = tk.LabelFrame(left_frame, text="Parameters", padx=10, pady=10)
        param_frame.pack(fill=tk.X, pady=5)
        
        animation_frame = tk.LabelFrame(left_frame, text="Animation", padx=10, pady=10)
        animation_frame.pack(fill=tk.X, pady=5)
        
        analysis_frame = tk.LabelFrame(left_frame, text="Analysis", padx=10, pady=10)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        # Display area
        self.fig = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control elements
        Button(control_frame, text="Load Image", command=self.load_image, width=20).pack(pady=5)
        Button(control_frame, text="Load DICOM", command=self.load_dicom, width=20).pack(pady=5)
        Button(control_frame, text="Generate Sinogram", command=self.generate_sinogram, width=20).pack(pady=5)
        Button(control_frame, text="Reconstruct Image", command=self.reconstruct_image, width=20).pack(pady=5)
        Button(control_frame, text="Calculate RMSE", command=self.calculate_and_show_rmse, width=20).pack(pady=5)
        Button(control_frame, text="Save as DICOM", command=self.save_dicom, width=20).pack(pady=5)
        Button(analysis_frame, text="Run RMSE Experiment", command=self.run_rmse_experiment, width=20).pack(pady=5)
        Button(analysis_frame, text="Show RMSE Plots", command=self.show_rmse_plots, width=20).pack(pady=5)


        # Parameters
        Label(param_frame, text="Number of Detectors:").pack(pady=(5, 0))
        self.detector_scale = Scale(param_frame, from_=90, to=720, orient=tk.HORIZONTAL, 
                                  resolution=90, command=self.update_detector_count)
        self.detector_scale.set(self.detector_count)
        self.detector_scale.pack(fill=tk.X)
        
        Label(param_frame, text="Angle Step (degrees):").pack(pady=(5, 0))
        self.angle_scale = Scale(param_frame, from_=0.5, to=5, orient=tk.HORIZONTAL, 
                               resolution=0.5, command=self.update_angle_step)
        self.angle_scale.set(self.angle_step)
        self.angle_scale.pack(fill=tk.X)
        
        Label(param_frame, text="Span Angle (degrees):").pack(pady=(5, 0))
        self.span_scale = Scale(param_frame, from_=45, to=270, orient=tk.HORIZONTAL, 
                              resolution=45, command=self.update_span_angle)
        self.span_scale.set(self.span_angle)
        self.span_scale.pack(fill=tk.X)
        
        self.filter_var = BooleanVar(value=self.use_filter)
        Checkbutton(param_frame, text="Use Filter", variable=self.filter_var,
                  command=self.toggle_filter).pack(pady=5)
        
        # Animation controls
        Label(animation_frame, text="Current Angle:").pack(pady=(5, 0))
        self.animation_scale = Scale(animation_frame, from_=0, to=180, orient=tk.HORIZONTAL,
                                   command=self.update_current_angle)
        self.animation_scale.config(to=self.span_angle)
        self.animation_scale.pack(fill=tk.X)
        
        animation_buttons = tk.Frame(animation_frame)
        animation_buttons.pack(pady=5)
        Button(animation_buttons, text="Start", command=self.start_animation).pack(side=tk.LEFT, padx=5)
        Button(animation_buttons, text="Stop", command=self.stop_animation).pack(side=tk.LEFT, padx=5)
        Button(animation_buttons, text="Reset", command=self.reset_animation).pack(side=tk.LEFT, padx=5)
        
        # Initial plot setup
        self.setup_plots()
    
    def setup_plots(self):
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        
        self.ax1.set_title("Original Image")
        self.ax2.set_title("Sinogram")
        self.ax3.set_title("Reconstructed Image")
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.bmp;*.tif"), ("All files", "*.*")]
        )
        if file_path:
            try:
                img = Image.open(file_path).convert('L')  # Convert to grayscale
                # Resize to manageable size if too large
                if max(img.size) > 500:
                    scale = 500 / max(img.size)
                    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                    img = img.resize(new_size, Image.LANCZOS)
                self.original_image = np.array(img)
                self.loaded_image_path = file_path 
                self.update_display()
                # Reset other data
                self.sinogram = None
                self.reconstructed_image = None
                self.current_sinogram = None
                self.current_reconstruction = None
                self.reset_animation()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def load_dicom(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
        )
        if file_path:
            try:
                ds = pydicom.dcmread(file_path)
                self.original_image = ds.pixel_array
                self.update_display()
                self.sinogram = None
                self.reconstructed_image = None
                self.current_sinogram = None
                self.current_reconstruction = None
                self.reset_animation()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading DICOM file: {str(e)}")

    def generate_sinogram(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            # Generate sinogram
            self.sinogram = radon_all(
                self.original_image, 
                int(180 / self.angle_step), 
                self.detector_count, 
                self.span_angle
            )
            self.update_display()
            self.reset_animation()
        except Exception as e:
            messagebox.showerror("Error", f"Error generating sinogram: {str(e)}")
    
    def reconstruct_image(self):
        if self.sinogram is None:
            messagebox.showwarning("Warning", "Please generate sinogram first")
            return
        
        try:
            self.reconstructed_image = inverse_radon_all(
                self.original_image.shape, 
                self.sinogram, 
                self.span_angle,
                use_filter = self.use_filter
            )
            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Error reconstructing image: {str(e)}")

    def save_dicom(self):
        if self.reconstructed_image is None:
            messagebox.showwarning("Warning", "No reconstructed image to save as DICOM.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".dcm",
            filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
        )

        if not file_path:
            return

        def on_patient_info_collected(patient_info):
            try:
                save_as_dicom(self.reconstructed_image, file_path, patient_info)
                messagebox.showinfo("Success", "DICOM file saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving DICOM file: {str(e)}")

        self.get_patient_info(on_patient_info_collected)


    def get_patient_info(self, callback):
        info_window = Toplevel(self.root)
        info_window.title("Patient Information")

        Label(info_window, text="Patient Name:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        name_entry = Entry(info_window)
        name_entry.grid(row=0, column=1, padx=5, pady=5)

        Label(info_window, text="Patient ID:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        id_entry = Entry(info_window)
        id_entry.grid(row=1, column=1, padx=5, pady=5)

        Label(info_window, text="Birthdate (YYYYMMDD):").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        birth_entry = Entry(info_window)
        birth_entry.grid(row=2, column=1, padx=5, pady=5)

        Label(info_window, text="Study Description:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        desc_entry = Entry(info_window)
        desc_entry.grid(row=3, column=1, padx=5, pady=5)

        def submit_info():
            patient_info = {
                'name': name_entry.get(),
                'id': id_entry.get(),
                'birthdate': birth_entry.get(),
                'description': desc_entry.get()
            }
            callback(patient_info)  # <-- przekazujemy dane do zewnętrznej funkcji
            info_window.destroy()

        Button(info_window, text="Save", command=submit_info).grid(row=4, column=0, columnspan=2, pady=10)


    def update_display(self):
        self.fig.clear()
        
        # Display original image
        ax1 = self.fig.add_subplot(131)
        if self.original_image is not None:
            ax1.imshow(self.original_image, cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Display sinogram
        ax2 = self.fig.add_subplot(132)
        if self.current_sinogram is not None:
            ax2.imshow(self.current_sinogram, cmap='gray', aspect='auto')
        elif self.sinogram is not None:
            ax2.imshow(self.sinogram, cmap='gray', aspect='auto')
        ax2.set_title("Sinogram")
        ax2.axis('off')
        
        # Display reconstructed image
        ax3 = self.fig.add_subplot(133)
        if self.current_reconstruction is not None:
            ax3.imshow(self.current_reconstruction, cmap='gray')
        elif self.reconstructed_image is not None:
            ax3.imshow(self.reconstructed_image, cmap='gray')
        ax3.set_title("Reconstructed Image")
        ax3.axis('off')

        
        self.fig.tight_layout()
        self.canvas.draw()


    # Parameter update callbacks
    def update_detector_count(self, value):
        self.detector_count = int(float(value))
    
    def update_angle_step(self, value):
        self.angle_step = float(value)
    
    def update_span_angle(self, value):
        self.span_angle = float(value)
        self.animation_scale.config(to=self.span_angle)
        self.animation_scale.set(0)
        self.current_angle = 0
    
    def toggle_filter(self):
        self.use_filter = self.filter_var.get()
    
    def calculate_and_show_rmse(self):
        if self.original_image is None or self.reconstructed_image is None:
            messagebox.showwarning("Warning", "Both original and reconstructed images are required")
            return
        
        rmse = calculate_rmse(self.original_image, self.reconstructed_image)
        messagebox.showinfo("RMSE", f"Root Mean Square Error: {rmse:.4f}")
    
    def start_animation(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        # Precompute full sinogram once
        self.full_sinogram = radon_all(
            self.original_image,
            int(180 / self.angle_step),
            self.detector_count,
            self.span_angle
        )

        self.is_animation_running = True
        self.animate()

    
    def stop_animation(self):
        self.is_animation_running = False
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
    
    def reset_animation(self):
        self.stop_animation()
        self.current_angle = 0
        self.animation_scale.set(0)
        self.current_sinogram = None
        self.current_reconstruction = None
        self.full_sinogram = None
        self.update_display()
    
    def animate(self):
        if not self.is_animation_running:
            return
        
        # Update angle
        self.current_angle += self.angle_step
        if self.current_angle >= self.span_angle:
            self.current_angle = 0
        
        self.animation_scale.set(self.current_angle)
        self.update_animation_frame()
        
        # Schedule next frame
        self.animation_id = self.root.after(100, self.animate)
    
    def update_animation_frame(self):
        if self.original_image is None or self.full_sinogram is None:
            return  # Safety check

        # Oblicz indeks w sinogramie zgodny z bieżącym kątem animacji
        current_idx = int(self.current_angle / self.angle_step)
        current_idx = min(current_idx, self.full_sinogram.shape[1] - 1)

        # Użyj wcześniej obliczonego pełnego sinogramu
        self.current_sinogram = self.full_sinogram[:, :current_idx + 1]

        # Rekonstrukcja z częściowego sinogramu
        self.current_reconstruction = inverse_radon_all(
            self.original_image.shape,
            self.current_sinogram,
            self.span_angle
        )

        self.update_display()

    
    def update_current_angle(self, value):
        self.current_angle = float(value)
        self.update_animation_frame()


    def run_rmse_experiment(self):
        try:
            if self.loaded_image_path:
                # Użyj sys.executable, aby uzyskać ścieżkę do interpretera Python
                python_exe = sys.executable
                
                # Użyj os.path.join zamiast ręcznego tworzenia ścieżek
                current_dir = os.path.dirname(os.path.abspath(__file__))
                experiments_path = os.path.join(current_dir, "experiments.py")
                
                # Użyj listy argumentów zamiast ciągu znaków
                args = [python_exe, experiments_path, self.loaded_image_path]
                
                subprocess.Popen(args)
                
                messagebox.showinfo("Success", "RMSE in progress. Check the 'results' folder soon.")
            else:
                messagebox.showwarning("Warning", "Please load an image first.")
        except Exception as e:
            messagebox.showerror("Error", f"Error running experiment: {str(e)}")


    def show_rmse_plots(self):
        try:
            # Otwórz folder wynikowy w eksploratorze plików lub przeglądarce
            results_folder = os.path.abspath("results")
            webbrowser.open(f"file:///{results_folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Error opening results folder: {str(e)}")
