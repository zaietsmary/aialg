from taipy.gui import Gui
from PIL import Image
import io
import base64
import numpy as np
import os
import pandas as pd

uploaded_files = []
selected_class = "Class A"
file_info = ""
classification_result = ""
test_image = None
test_vector_abs = []
test_vector_norm = []
recognition_result = ""
predicted_class = ""
test_table_data = pd.DataFrame(columns=["Тип", "Значення"])
cluster_stats_table = pd.DataFrame(columns=["Клас", "Мінімум", "Максимум", "Середнє", "Дисперсія"])
class_a_images = []
class_b_images = []
table_data_a = pd.DataFrame(columns=["Зразок", "Файл", "Абсолютний вектор", "Нормований вектор"])
table_data_b = pd.DataFrame(columns=["Зразок", "Файл", "Абсолютний вектор", "Нормований вектор"])
training_data = {
    "Class A": [],
    "Class B": []
}
training_stats = {
    "Class A": {},
    "Class B": {}
}
w = np.random.rand(4*5).tolist()
b = 0
r = 0.1

def vector_calculation(img, grid_rows: int = 4, grid_cols: int = 5):
    pixels = np.array(img.convert("L"))
    pixels = np.where(pixels > 128, 255, 0)
    h, w = pixels.shape
    seg_h = h // grid_rows
    seg_w = w // grid_cols
    vector = [
        np.sum(pixels[i * seg_h:min((i + 1) * seg_h, h), j * seg_w:min((j + 1) * seg_w, w)] == 0)
        for i in range(grid_rows) for j in range(grid_cols)
    ]
    total = sum(vector) or 1
    normalized = [round(float(v / total), 4) for v in vector]
    return [int(x) for x in vector], normalized

def update_table_data(state):
    state.table_data_a = create_table_data(training_data["Class A"])
    state.table_data_b = create_table_data(training_data["Class B"])

def create_table_data(class_data):
    if not class_data:
        return pd.DataFrame(columns=["Зразок", "Файл", "Абсолютний вектор", "Нормований вектор"])
    data = []
    for i, item in enumerate(class_data):
        data.append({
            "Зразок": f"#{i + 1}",
            "Файл": item["filename"],
            "Абсолютний вектор": str(item["vector_abs"]),
            "Нормований вектор": str(item["vector_norm"])
        })
    return pd.DataFrame(data)

def update_test_table(state, vector_abs, vector_norm):
    data = [
        {"Тип": "Абсолютний вектор", "Значення": str(vector_abs)},
        {"Тип": "Нормований вектор", "Значення": str(vector_norm)}
    ]
    state.test_table_data = pd.DataFrame(data)

def calculate_cluster_stats():
    for class_name in ["Class A", "Class B"]:
        images_data = training_data[class_name]
        if images_data:
            vectors = np.array([img_data["vector_norm"] for img_data in images_data])
            training_stats[class_name] = {
                "min": vectors.min(axis=0),
                "max": vectors.max(axis=0),
                "mean": vectors.mean(axis=0),
                "variance": vectors.var(axis=0)
            }

def update_cluster_stats(state):
    cluster_stats = []
    for class_name in ["Class A", "Class B"]:
        if class_name in training_stats:
            stats = training_stats[class_name]
            if stats:
                cluster_stats.append({
                    "Клас": class_name,
                    "Мінімум": str([round(x, 4) for x in stats["min"]]),
                    "Максимум": str([round(x, 4) for x in stats["max"]]),
                    "Середнє": str([round(x, 4) for x in stats["mean"]]),
                    "Дисперсія": str([round(x, 4) for x in stats["variance"]])
                })
    state.cluster_stats_table = pd.DataFrame(cluster_stats)

def on_files_upload(state):
    try:
        if not state.uploaded_files:
            return

        state.file_info = "Обробка файлів..."
        state.classification_result = ""

        current_class = state.selected_class
        current_count = len(training_data[current_class])
        available_slots = 10 - current_count

        if available_slots <= 0:
            state.file_info = f"⚠️ У {current_class} вже є 10 зображень!"
            return

        files_to_process = state.uploaded_files[:available_slots]
        processed_count = 0

        for file_path in files_to_process:
            file_name = os.path.basename(file_path)
            with open(file_path, 'rb') as f:
                img_data = f.read()
            img = Image.open(io.BytesIO(img_data))
            img_gray = img.convert("L")

            vector, normalized = vector_calculation(img_gray)

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

            training_data[current_class].append({
                "image": img_b64,
                "vector_abs": vector,
                "vector_norm": normalized,
                "filename": file_name
            })

            processed_count += 1

        state.class_a_images = [img["image"] for img in training_data["Class A"]]
        state.class_b_images = [img["image"] for img in training_data["Class B"]]
        update_table_data(state)
        calculate_cluster_stats()
        update_cluster_stats(state)
        train_perceptron(state)

        state.file_info = f"✅ Додано {processed_count} фото до {current_class}"
        state.classification_result = f"Тепер у {current_class}: {len(training_data[current_class])}/10 зображень"
        state.uploaded_files = []

    except Exception as e:
        state.file_info = f"Помилка: {str(e)}"
        state.classification_result = ""

def on_test_file_upload(state):
    try:
        if state.test_image:
            file_name = os.path.basename(state.test_image)
            with open(state.test_image, 'rb') as f:
                img_data = f.read()
                img = Image.open(io.BytesIO(img_data))
                img_gray = img.convert("L")
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
                vector, normalized = vector_calculation(img_gray)

                state.test_image = img_b64
                state.test_vector_abs = vector
                state.test_vector_norm = normalized
                update_test_table(state, vector, normalized)
                state.recognition_result = f"Фото завантажено для розпізнавання! Розмір: {img.size}"
                state.predicted_class = "Очікує розпізнавання..."

    except Exception as e:
        state.recognition_result = f"Помилка: {str(e)}"

def d(weight):
    if weight > 0:
        return 1
    return -1

def train_perceptron(state):
    if not hasattr(state, "w"):
        state.w = [np.random.rand() for _ in range(4 * 5)]
        state.b = 0

    if len(training_data["Class A"]) < 10 or len(training_data["Class B"]) < 10:
        state.classification_result = "⚠️ Недостатньо даних для навчання"
        return

    classes_list = ["Class A", "Class B"]
    for _ in range(100):
        for class_name in classes_list:
            for sample in training_data[class_name]:
                x = sample["vector_norm"]
                desired = 1 if class_name == "Class A" else -1
                weight_sum = sum(xi * wi for xi, wi in zip(x, state.w)) + state.b
                y = d(weight_sum)
                if y != desired:
                    state.w = [wi + r * (desired - y) * xi for wi, xi in zip(state.w, x)]
                    state.b = state.b + r * (desired - y)

def recognize_image(state):
    if not state.test_vector_norm:
        state.recognition_result = "⚠️ Спочатку завантажте фото для розпізнавання"
        return

    x_test = np.array(state.test_vector_norm)
    weight_sum = np.dot(x_test, state.w) + state.b
    y = d(weight_sum)

    state.predicted_class = "Class A" if y == 1 else "Class B"
    state.recognition_result = f"📌 Розпізнано як: {state.predicted_class}"

page = """
<|layout|columns=1 1|gap=15px|>

## Вибір класу:
<|{selected_class}|selector|lov=Class A;Class B|label=Оберіть клас|dropdown|width=100%|>

## Завантаження фото (10 на клас):
<|{uploaded_files}|file_selector|extensions=.png,.jpg,.jpeg|on_action=on_files_upload|label=Завантажте фото|multiple|width=100%|>
<|{file_info}|text|>
<|{classification_result}|text|>

## Завантаження для класифікації:
<|{test_image}|file_selector|extensions=.png,.jpg,.jpeg|on_action=on_test_file_upload|label=Завантажте тестове фото|width=100%|>
<|{test_image}|image|height=200px|width=200px|>
<|button|label=Розпізнати|on_action=recognize_image|>
<|{recognition_result}|text|>

## Вектори ознак тестового зображення:
<|{test_table_data}|table|page_size=2|width=100%|>

## Статистика кластерів:
<|{cluster_stats_table}|table|page_size=3|width=100%|>

## Галерея зображень:
### Class A
<|layout|columns=1 1 1 1 1|gap=15px|>
<|{class_a_images[0] if len(class_a_images) > 0 else None}|image|height=120px|width=120px|>
<|{class_a_images[1] if len(class_a_images) > 1 else None}|image|height=120px|width=120px|>
<|{class_a_images[2] if len(class_a_images) > 2 else None}|image|height=120px|width=120px|>
<|{class_a_images[3] if len(class_a_images) > 3 else None}|image|height=120px|width=120px|>
<|{class_a_images[4] if len(class_a_images) > 4 else None}|image|height=120px|width=120px|>
<|layout|columns=1 1 1 1 1|gap=15px|>
<|{class_a_images[5] if len(class_a_images) > 5 else None}|image|height=120px|width=120px|>
<|{class_a_images[6] if len(class_a_images) > 6 else None}|image|height=120px|width=120px|>
<|{class_a_images[7] if len(class_a_images) > 7 else None}|image|height=120px|width=120px|>
<|{class_a_images[8] if len(class_a_images) > 8 else None}|image|height=120px|width=120px|>
<|{class_a_images[9] if len(class_a_images) > 9 else None}|image|height=120px|width=120px|>
<|{table_data_a}|table|page_size=5|width=100%|>

### Class B
<|layout|columns=1 1 1 1 1|gap=15px|>
<|{class_b_images[0] if len(class_b_images) > 0 else None}|image|height=120px|width=120px|>
<|{class_b_images[1] if len(class_b_images) > 1 else None}|image|height=120px|width=120px|>
<|{class_b_images[2] if len(class_b_images) > 2 else None}|image|height=120px|width=120px|>
<|{class_b_images[3] if len(class_b_images) > 3 else None}|image|height=120px|width=120px|>
<|{class_b_images[4] if len(class_b_images) > 4 else None}|image|height=120px|width=120px|>
<|layout|columns=1 1 1 1 1|gap=15px|>
<|{class_b_images[5] if len(class_b_images) > 5 else None}|image|height=120px|width=120px|>
<|{class_b_images[6] if len(class_b_images) > 6 else None}|image|height=120px|width=120px|>
<|{class_b_images[7] if len(class_b_images) > 7 else None}|image|height=120px|width=120px|>
<|{class_b_images[8] if len(class_b_images) > 8 else None}|image|height=120px|width=120px|>
<|{class_b_images[9] if len(class_b_images) > 9 else None}|image|height=120px|width=120px|>
<|{table_data_b}|table|page_size=5|width=100%|>

"""

if __name__ == "__main__":
    gui = Gui(page)
    gui.run(debug=True, port=5000)

