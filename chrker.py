import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class DigitAugmentor:
    def __init__(self):
        self.augmentation_pipeline = A.Compose([
            # Геометрические преобразования
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-5, 5),
                shear=(-2, 2),
                p=0.7
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Искажения
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            A.OpticalDistortion(distort_limit=0.05, p=0.2),
            
            # Шум и качество изображения
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.4),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
            A.Blur(blur_limit=3, p=0.4),
            A.MotionBlur(blur_limit=5, p=0.3),
            
            # Яркость/контрастность
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.6
            ),
            A.CLAHE(clip_limit=2.0, p=0.3),
            
            # Цветовые преобразования
            A.HueSaturationValue(
                hue_shift_limit=(-5, 5),
                sat_shift_limit=(-10, 10),
                val_shift_limit=(-10, 10),
                p=0.4
            ),
            
            # Морфологические операции (используем правильные названия)
            A.augmentations.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, p=0.2),
        ])
    
    def augment_image(self, image_path, output_path):
        """Применяет аугментацию к изображению и сохраняет результат"""
        try:
            # Загрузка изображения
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Не удалось загрузить изображение: {image_path}")
                return
            
            # Конвертируем в RGB для albumentations (некоторые трансформации требуют RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Применение аугментации
            augmented = self.augmentation_pipeline(image=image_rgb)
            augmented_image = augmented['image']
            
            # Конвертируем обратно в grayscale
            augmented_gray = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2GRAY)
            
            # Дополнительные аугментации через PIL для разнообразия
            pil_image = Image.fromarray(augmented_gray)
            
            # Случайные дополнительные преобразования
            if random.random() < 0.3:
                # Изменение резкости
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            if random.random() < 0.2:
                # Легкое размытие
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
            
            # Сохранение результата
            pil_image.save(output_path)
            print(f"Аугментировано: {image_path} -> {output_path}")
            
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()

def process_folder(folder_path, num_augmentations=3):
    """Обрабатывает все изображения в папке"""
    augmentor = DigitAugmentor()
    
    # Поддерживаемые форматы изображений
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # Создаем папку для аугментированных изображений, если её нет
    aug_folder = os.path.join(folder_path, 'augmented')
    os.makedirs(aug_folder, exist_ok=True)
    
    # Получаем список файлов
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(supported_formats) and not f.endswith('_a.png')]
    
    print(f"Найдено {len(image_files)} изображений для аугментации")
    
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        
        # Создаем несколько аугментированных версий
        for i in range(num_augmentations):
            # Генерируем имя для аугментированного файла
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_a{i+1}{ext}"
            output_path = os.path.join(aug_folder, output_filename)
            
            # Пропускаем если файл уже существует
            if os.path.exists(output_path):
                print(f"Файл уже существует: {output_path}")
                continue
                
            # Применяем аугментацию
            augmentor.augment_image(image_path, output_path)

def augment_single_image(image_path, output_suffix='_a'):
    """Аугментирует одно изображение и сохраняет в той же папке"""
    augmentor = DigitAugmentor()
    
    # Генерируем выходное имя
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}{output_suffix}{ext}"
    output_path = os.path.join(directory, output_filename)
    
    augmentor.augment_image(image_path, output_path)
    return output_path

def simple_augmentation_pipeline():
    """Упрощенный пайплайн аугментации без ошибок"""
    return A.Compose([
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.03, 0.03),
            rotate=(-3, 3),
            p=0.6
        ),
        A.GaussNoise(var_limit=10.0, p=0.4),
        A.Blur(blur_limit=3, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.15, 0.15),
            contrast_limit=(-0.15, 0.15),
            p=0.5
        ),
    ])

# Альтернативная версия с упрощенной аугментацией
class SimpleDigitAugmentor:
    def __init__(self):
        self.pipeline = simple_augmentation_pipeline()
    
    def augment_image(self, image_path, output_path):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return
            
            augmented = self.pipeline(image=image)
            augmented_image = augmented['image']
            
            cv2.imwrite(output_path, augmented_image)
            print(f"Аугментировано: {output_path}")
            
        except Exception as e:
            print(f"Ошибка: {e}")

# Пример использования
if __name__ == "__main__":
    # Установите albumentations если нет: pip install albumentations
    
    # 1. Обработка всей папки
    folder_path = "data/train/images"  # Замените на ваш путь
    process_folder(folder_path, num_augmentations=3)
    