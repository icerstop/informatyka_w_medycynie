import tkinter as tk
from tkinter import filedialog, Scale, messagebox, Toplevel, Entry, Label, Button, Checkbutton, BooleanVar
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from algorithms import radon_all, inverse_radon_all, calculate_rmse

class CTScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CT Scanner Simulation")
        
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
        Button(control_frame, text="Generate Sinogram", command=self.generate_sinogram, width=20).pack(pady=5)
        Button(control_frame, text="Reconstruct Image", command=self.reconstruct_image, width=20).pack(pady=5)
        Button(control_frame, text="Calculate RMSE", command=self.calculate_and_show_rmse, width=20).pack(pady=5)
        
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
                self.update_display()
                # Reset other data
                self.sinogram = None
                self.reconstructed_image = None
                self.current_sinogram = None
                self.current_reconstruction = None
                self.reset_animation()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
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
                self.span_angle
            )
            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Error reconstructing image: {str(e)}")
    
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
        if self.sinogram is not None:
            ax2.imshow(self.sinogram, cmap='gray', aspect='auto')
        ax2.set_title("Sinogram")
        ax2.axis('off')
        
        # Display reconstructed image
        ax3 = self.fig.add_subplot(133)
        if self.reconstructed_image is not None:
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
        self.update_display()
    
    def animate(self):
        if not self.is_animation_running:
            return
        
        # Update angle
        self.current_angle += self.angle_step
        if self.current_angle >= 180:
            self.current_angle = 0
        
        self.animation_scale.set(self.current_angle)
        self.update_animation_frame()
        
        # Schedule next frame
        self.animation_id = self.root.after(100, self.animate)
    
    def update_animation_frame(self):
        if self.original_image is None:
            return
        
        # Calculate angles up to current
        current_idx = int(min(self.current_angle / self.angle_step, 180 / self.angle_step - 1))
        current_angles = np.arange(0, self.current_angle + self.angle_step, self.angle_step)
        
        # Generate partial sinogram
        self.current_sinogram = radon_all(
            self.original_image, 
            len(current_angles), 
            self.detector_count, 
            self.span_angle
        )
        
        # Reconstruct partial image
        self.current_reconstruction = inverse_radon_all(
            self.original_image.shape, 
            self.current_sinogram, 
            self.span_angle
        )
        
        self.update_display()
    
    def update_current_angle(self, value):
        self.current_angle = float(value)
        self.update_animation_frame()