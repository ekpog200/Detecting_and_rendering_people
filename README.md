## Описание проекта

В проекте реализована задача распознования объектов (людей) на видео, с использованием библиотеки OpenCV

## Workflow
Процесс работы с данными в проекте разделён на несколько этапов:
### Сбор данных
В папке input_video помещается видео, на котором необходимо провести распознование объектов. Видео должно называться crowd.mp4 (или же название можно поменять после в файле)
В папку input_data требуется положить готовые веса yolov4 (yolov4.weights), а также cfg файл (yolov4.cfg). Данные можно получить с https://github.com/AlexeyAB/darknet/releases
### Распознование
Основной код программы (точка входа) находится в файле low_model/low_video_recognizer.py. В файле происходит загрузка всех данных, обработка, распознование, запись результатов.
### Вывод результата
В результате работы программы, видео будет размещено в папку output_video

## Документация к настройке и запуску
Настройка:
1. Установите виртуальное окружение и активируйте его
2. Установите требуемые библиотеки для работы (low_model/requirements.txt)
3. При необходимости поменяйте в файле low_model/low_video_recognizer.py пути к весам yolov4 (переменная net), путь к видео, которое необходимо распознать (переменная cap), путь к папке результата (output_video)
4. Поместить веса yolov4 (получить можно здесь https://github.com/AlexeyAB/darknet/releases) в папку input_data
5. Поместить видео, которое требуется распознать, в input_video

## Результаты
В результате получилось обработать видео и корректно распознать людей на нём. Для удобства, итоговое распознанное видео поместил на яндекс диск https://disk.yandex.ru/i/fr_7Xs4q2veaAA . 

## Рекомендации к улучшению
1. Использование Docker для быстрого развертывания / масштабирования
2. Использование кэширования для ускорения работы, т.к. в данный момент происходит подлагивание при распозновании объектов
3. Включение логгирования и мониторинга (например mlflow)
4. Дообучение готовой модели на своих данных, для лучшего обнаружения объектов
5. Написание тестов для корректной работы и уменьшения фактора ошибок
