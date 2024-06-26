import cv2
import numpy as np
import os
from input_download import download_input


def main() -> None:
    # Загрузка предобученной модели YOLO + видео
    folder_weight = "../input_data/"
    folder_path = "../input_video/"
    files = download_input(folder_weight, folder_path)

    # загрузка весов в dnn
    net = cv2.dnn.readNet("../input_data/yolov4.weights", "../input_data/yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    for file_name in files:
        # берём каждый файл в каталоге и обрабатываем его
        video_path = os.path.join(folder_path, file_name)
        cap = cv2.VideoCapture(video_path)

        # Вывод видео в output_video
        size = (int(cap.get(3)), int(cap.get(4)))
        fps = cap.get(5)
        output_name = '../output_video/' + file_name[:-4] + '.mp4'
        output_video = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Преобразование изображения для входа в yolo, передача на вход
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Информация о найденных объектах
            class_ids, confidences, boxes = [], [], []

            for out in outs:
                for detection in out:
                    # Берём вероятности класса, берём из них наибольший, вытягиваем наибольший класс
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.4 and class_id == 0:
                        # Выбираем уверенность и 0й класс = Person в Yolov4
                        # Преобразование нормализованных координат и размеров в реальные
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        w = int(detection[2] * frame.shape[1])
                        h = int(detection[3] * frame.shape[0])
                        # Вычисление верхнего левого угла объекта
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # подавления немаксимальных значений для ограничивающих прямоугольников (0,3) + порог на уверенность (0,3)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

            for i in range(len(boxes)):
                if i in indexes:
                    # Цикл для обозначения людей + вывод текста и уверенности
                    x, y, w, h = boxes[i]
                    label = f"Person {confidences[i]:.2f}"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

            output_video.write(frame)

            # Для корректного отображения видео
            frame_min = frame.copy()
            cv2.imshow(file_name[:-4] + ' detection', frame_min)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()