import os
import shutil

# -----------------------
# Настройки
# -----------------------
src_dir = "data/val/images"       # исходные изображения
dst_dir = "data/val/images_aug"   # куда сохранять с blank

os.makedirs(dst_dir, exist_ok=True)

ALPHABET = "0123456789"
BLANK = "-"  # blank для CTC

# -----------------------
# Обработка
# -----------------------
files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith((".jpg",".png",".jpeg"))])
for f in files:
    name, ext = os.path.splitext(f)
    label = name.split("_")[0]

    # пропускаем файлы с некорректными символами
    if not all(c in ALPHABET for c in label):
        continue

    # вставляем blank между повторяющимися символами
    new_label = []
    prev = None
    for c in label:
        if prev is not None and c == prev:
            new_label.append(BLANK)
        new_label.append(c)
        prev = c

    new_name = "".join(new_label) + "_" + "_".join(name.split("_")[1:]) + ext
    src_path = os.path.join(src_dir, f)
    dst_path = os.path.join(dst_dir, new_name)

    shutil.copy(src_path, dst_path)

print(f"✅ Обработано {len(files)} файлов, сохранено в {dst_dir}")
