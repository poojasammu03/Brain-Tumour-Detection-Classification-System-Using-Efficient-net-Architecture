# brain_tumor_ui.py
# Separate UI application to load saved model and test images

import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BrainTumorUI:
    def __init__(self):
        self.model = None
        self.img_size = 224
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.classification_type = 'multiclass'
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        self.root = tk.Tk()
        self.root.title("Brain Tumor Detection - Image Tester")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f8ff')
        
        # Main title
        title_label = tk.Label(self.root, text="🧠 Brain Tumor Detection System", 
                              font=("Arial", 24, "bold"), bg='#f0f8ff', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # Model status frame
        status_frame = tk.Frame(self.root, bg='#f0f8ff')
        status_frame.pack(pady=15)
        
        tk.Label(status_frame, text="Model Status:", font=("Arial", 12, "bold"), 
                bg='#f0f8ff', fg='#34495e').pack()
        
        self.status_label = tk.Label(status_frame, text="❌ Not Loaded", 
                                    font=("Arial", 12), bg='#f0f8ff', fg='#e74c3c')
        self.status_label.pack(pady=5)
        
        # Load model button
        load_button = tk.Button(status_frame, text="🔄 Load Trained Model", 
                               command=self.load_model, font=("Arial", 11, "bold"),
                               bg='#3498db', fg='white', padx=20, pady=8,
                               relief='raised', borderwidth=2)
        load_button.pack(pady=10)
        
        # Separator
        separator1 = ttk.Separator(self.root, orient='horizontal')
        separator1.pack(fill='x', padx=50, pady=20)
        
        # Image selection frame
        select_frame = tk.Frame(self.root, bg='#f0f8ff')
        select_frame.pack(pady=15)
        
        tk.Label(select_frame, text="📁 Select MRI Image for Analysis", 
                font=("Arial", 14, "bold"), bg='#f0f8ff', fg='#34495e').pack()
        
        self.select_button = tk.Button(select_frame, text="Choose Image", 
                                      command=self.select_image, state='disabled',
                                      font=("Arial", 11), bg='#27ae60', fg='white',
                                      padx=20, pady=8, relief='raised', borderwidth=2)
        self.select_button.pack(pady=10)
        
        # Image display frame
        self.image_frame = tk.Frame(self.root, bg='#f0f8ff')
        self.image_frame.pack(pady=20)
        
        self.image_label = tk.Label(self.image_frame, text="No image selected\n\n📷", 
                                   bg='white', fg='#7f8c8d', width=40, height=15,
                                   font=("Arial", 12), relief='sunken', borderwidth=2)
        self.image_label.pack()
        
        # Predict button
        self.predict_button = tk.Button(self.image_frame, text="🔍 Analyze Image", 
                                       command=self.predict_image, state='disabled',
                                       font=("Arial", 12, "bold"), bg='#e67e22', fg='white',
                                       padx=30, pady=10, relief='raised', borderwidth=2)
        self.predict_button.pack(pady=15)
        
        # Results frame
        results_frame = tk.Frame(self.root, bg='#ecf0f1', relief='ridge', borderwidth=2)
        results_frame.pack(pady=20, padx=40, fill='x')
        
        tk.Label(results_frame, text="📊 Analysis Results", 
                font=("Arial", 14, "bold"), bg='#ecf0f1', fg='#2c3e50').pack(pady=10)
        
        self.result_label = tk.Label(results_frame, text="", 
                                    font=("Arial", 16, "bold"), bg='#ecf0f1')
        self.result_label.pack(pady=5)
        
        self.confidence_label = tk.Label(results_frame, text="", 
                                        font=("Arial", 12), bg='#ecf0f1', fg='#7f8c8d')
        self.confidence_label.pack(pady=(0, 8))
        
        # Instructions
        instructions = tk.Label(self.root, 
                               text="Instructions: 1) Load the trained model  2) Select an MRI image  3) Click Analyze",
                               font=("Arial", 10), bg='#f0f8ff', fg='#7f8c8d')
        instructions.pack(side='bottom', pady=10)
        
    def load_model(self):
        """Load the trained model"""
        try:
            # Check if model files exist
            model_path = 'best_brain_tumor_model.keras'
            config_path = 'model_config.json'
            
            if not os.path.exists(model_path):
                messagebox.showerror("Error", 
                    f"Model file '{model_path}' not found!\n\nPlease make sure you have:\n"
                    "1. Trained the model first\n"
                    "2. The model file is in the same directory as this script")
                return
            
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load configuration if exists
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.img_size = config.get('img_size', 224)
                    self.class_names = config.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])
                    self.classification_type = config.get('classification_type', 'multiclass')
            
            # Update UI
            self.status_label.config(text="✅ Model Loaded Successfully", fg='#27ae60')
            self.select_button.config(state='normal')
            
            messagebox.showinfo("Success", 
                f"Model loaded successfully!\n\n"
                f"Type: {self.classification_type}\n"
                f"Classes: {', '.join(self.class_names)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def select_image(self):
        """Select an image file for testing"""
        file_path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.display_image(file_path)
            self.predict_button.config(state='normal')
            self.result_label.config(text="")
            self.confidence_label.config(text="")
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            # Open and resize image for display
            img = Image.open(image_path)
            img.thumbnail((350, 350), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Show image path
            filename = os.path.basename(image_path)
            self.image_label.config(text=f"Selected: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image:\n{str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction (same as training)"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_image(self):
        """Predict the selected image"""
        try:
            if not hasattr(self, 'selected_image_path'):
                messagebox.showerror("Error", "No image selected!")
                return
                
            if self.model is None:
                messagebox.showerror("Error", "No model loaded!")
                return
            
            # Preprocess image
            processed_img = self.preprocess_image(self.selected_image_path)
            if processed_img is None:
                messagebox.showerror("Error", "Failed to preprocess image!")
                return
            
            # Make prediction
            prediction = self.model.predict(processed_img, verbose=0)
            
            if self.classification_type == 'binary':
                prob = prediction[0][0]
                pred_class = "Tumor" if prob > 0.5 else "No Tumor"
                confidence = prob if prob > 0.5 else 1 - prob
            else:
                pred_idx = np.argmax(prediction)
                pred_class = self.class_names[pred_idx]
                confidence = prediction[0][pred_idx]
            
            # Display results with emojis and colors
            result_text = f"🎯 Prediction: {pred_class.upper()}"
            confidence_text = f"📈 Confidence: {confidence:.4f} ({confidence*100:.2f}%)"
            
            self.result_label.config(text=result_text)
            self.confidence_label.config(text=confidence_text)
            
            # Color code the result
            if pred_class.lower() in ['notumor', 'no tumor']:
                self.result_label.config(fg='#27ae60')  # Green for no tumor
                result_emoji = "✅ Good News!"
            else:
                self.result_label.config(fg='#e74c3c')  # Red for tumor detected
                result_emoji = "⚠️ Tumor Detected"
            
            # Show popup with result
            messagebox.showinfo("Analysis Complete", 
                f"{result_emoji}\n\n"
                f"Prediction: {pred_class}\n"
                f"Confidence: {confidence*100:.2f}%\n\n"
                f"Note: This is for educational purposes only.\n"
                f"Please consult a medical professional for proper diagnosis.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
    
    def run(self):
        """Run the UI application"""
        print("🚀 Starting Brain Tumor Detection UI...")
        print("📁 Make sure 'best_brain_tumor_model.keras' is in the current directory")
        self.root.mainloop()

def main():
    """Main function to run the UI application"""
    print("="*60)
    print("🧠 BRAIN TUMOR DETECTION - UI APPLICATION")
    print("="*60)
    print()
    
    # Check if model exists
    if os.path.exists('best_brain_tumor_model.keras'):
        print("✅ Model file found!")
    else:
        print("❌ Model file 'best_brain_tumor_model.keras' not found!")
        print("   Please train the model first using the main training script.")
        print()
    
    if os.path.exists('model_config.json'):
        print("✅ Configuration file found!")
    else:
        print("⚠️  Configuration file not found, using defaults.")
    
    print()
    print("🎯 Starting UI application...")
    
    # Create and run UI
    app = BrainTumorUI()
    app.run()

if __name__ == "__main__":
    main()