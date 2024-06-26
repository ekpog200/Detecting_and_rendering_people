import requests
import os


def download_input(folder_weight, folder_path) -> list[str]:
    folder_weight = "../input_data/"
    files_weight = [f for f in os.listdir(folder_weight) if os.path.isfile(os.path.join(folder_weight, f))]
    files_weight.remove('.gitkeep')
    if ('yolov4.weights' not in files_weight) or ('yolov4.cfg' not in files_weight):
        print("Скачивание yolov4 весов. Ожидайте (245МБ)")
        if 'yolov4.weights' not in files_weight:
            url = 'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights'
            r = requests.get(url)
            if r.status_code == 200:
                with open(folder_weight + 'yolov4.weights', 'wb') as file:
                    file.write(r.content)
            else:
                raise FileNotFoundError("Проверьте путь загрузки файла yolov4.weights")

        if 'yolov4.cfg' not in files_weight:
            url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg'
            rl = requests.get(url)
            if rl.status_code == 200:
                with open(folder_weight + 'yolov4.cfg', 'wb') as file:
                    file.write(rl.content)
            else:
                raise FileNotFoundError("Проверьте путь загрузки файла yolov4.cfg")
        print('Скачивание yolov4 весов закончено')

    # Чтение всех видео в input_video папке
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.remove('.gitkeep')
    if len(files) == 0:
        raise FileNotFoundError("Поместите хотя бы одно видео в input_video")

    return files
