## Описание проекта
В проекте реализована задача распознования объектов (людей) на видео, с использованием библиотеки OpenCV

## Результаты
В результате получилось обработать видео и корректно распознать людей на нём. Для удобства тестовое итоговое видео с распознанными людьми было размещено на диск https://disk.yandex.ru/i/fr_7Xs4q2veaAA . 

## Workflow
Процесс работы с данными в проекте разделён на несколько этапов:
### Сбор данных
В папке input_video помещается видео (любое кол-во, все буду обрабатываться последовательно), на котором необходимо провести распознование объектов.
В папку input_data требуется положить готовые веса yolov4 (yolov4.weights), а также cfg файл (yolov4.cfg). Если их нет, то они будут скачаны автоматически.
В файле input_download.py подготавливаются входные данные (веса, если их нет и обрабатываемые видео)
### Распознование
Основной код программы (точка входа) находится в файле low_model/low_video_recognizer.py. В файле происходит обработка, распознование, запись результатов.
### Вывод результата
Результирующее видео размещается в папку output_video

## Документация к настройке и запуску
Настройка:
1. Установить требуемые библиотеки для работы (low_model/requirements.txt).
2. Поместить видео, которое требуется распознать, в input_video
3. Запустить программу low_video_recognizer.py

## Рекомендации к улучшению
1. Использование Docker для быстрого развертывания / масштабирования
2. Использование кэширования / GPU для ускорения работы, т.к. в данный момент происходит подлагивание при распозновании объектов.
3. Включение логгирования и мониторинга (например mlflow)
4. Дообучение готовой модели на своих данных, для лучшего обнаружения объектов
5. Написание тестов для корректной работы и уменьшения фактора ошибок
