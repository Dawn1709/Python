import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
train_dir = './data'  # Thư mục chứa ảnh huấn luyện

# Kiểm tra thư mục có tồn tại không
if not os.path.exists(train_dir):
    print("⚠️ Không tìm thấy thư mục dữ liệu! Hãy đảm bảo bạn đã tải dataset vào thư mục 'data'.")
    exit()

# Chuẩn bị dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Định nghĩa mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

    # Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Huấn luyện mô hình
print(" Đang huấn luyện mô hình...")
model.fit(train_generator, epochs=10)

    # Lưu mô hình sau khi huấn luyện
model.save("fruit_model.h5")
print(" Mô hình đã được lưu thành công dưới dạng 'fruit_model.h5'!")
    # Định nghĩa danh sách nhãn cho các loại quả
labels = ["Quả táo", "Quả chuối", "Quả cherry", "Quả saboche", "Quả nho", "Quả kiwi", "Quả xoài", "Quả cam", "Quả dâu"]

    # Định nghĩa mô hình CNN 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

    # Kiểm tra nếu có file trọng số thì load, nếu không thì hiển thị cảnh báo
if os.path.exists("fruit_model.h5"):
    model.load_weights("fruit_model.h5")
else:
    print(" Chưa có mô hình được huấn luyện! Hãy huấn luyện mô hình trước.")

    # Cấu hình giao diện Tkinter
root = tk.Tk()
root.title("Ứng dụng Nhận diện Trái Cây")
root.geometry("500x600")
root.configure(bg="white")

    # Nhãn hiển thị ảnh và kết quả
label_img = tk.Label(root, bg="white")
label_img.pack(pady=20)
label_result = tk.Label(root, text="Kết quả: ", font=("Arial", 16), bg="white", fg="blue")
label_result.pack(pady=10)

def choose_image():
    """Chọn ảnh từ máy tính"""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    
    # Hiển thị ảnh lên giao diện
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    label_img.config(image=img_tk)
    label_img.image = img_tk

    # Nhận diện ảnh
    predict_image(file_path)

def predict_image(file_path):
    """Nhận diện ảnh và hiển thị kết quả"""
    test_image = load_img(file_path, target_size=(32, 32))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  

    # Dự đoán
    predictions = model.predict(test_image)
    predicted_class = np.argmax(predictions[0])
    result = labels[predicted_class]

    
    label_result.config(text=f"Kết quả: {result}")

   
btn_choose = tk.Button(root, text="Chọn Ảnh", command=choose_image, font=("Arial", 14), bg="green", fg="white")
btn_choose.pack(pady=20)


root.mainloop()
