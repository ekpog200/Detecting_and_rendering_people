import cv2
import numpy as np

# Загрузка предобученной модели YOLO и преобразование индексов
net = cv2.dnn.readNet("../input_data/yolov4.weights", "../input_data/yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Чтение видео
cap = cv2.VideoCapture("../input_video/crowd.mp4")

# Вывод видео в output1.avi
size = (int(cap.get(3)), int(cap.get(4)))
output_video = cv2.VideoWriter('../output_video/BNB_video2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while True:
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
    frame_min = cv2.resize(frame_min, (int(frame.shape[1] // 2), int(frame.shape[0] // 2)))

    cv2.imshow('BND_Test', frame_min)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
output_video.release()
cv2.destroyAllWindows()
