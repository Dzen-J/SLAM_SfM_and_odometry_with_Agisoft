import csv

def parse_cameras(file_path):
    cameras_data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Ищем строки с данными камеры, пропуская строки комментариев
        for line in lines:
            if line.startswith("#") or line.strip() == '':
                continue

            # Разделение строки на отдельные значения
            parts = line.strip().split()

            if len(parts) != 13:
                print(f"Пропущена строка из-за некорректного формата: {line.strip()}")
                continue

            try:
                # Парсинг матрицы поворота (9 элементов) и вектора трансляции (3 элемента)
                rotation_matrix = list(map(float, parts[:9]))
                translation_vector = list(map(float, parts[9:12]))
                focal_length = float(parts[12])

                cameras_data.append({
                    'R00': rotation_matrix[0], 'R01': rotation_matrix[1], 'R02': rotation_matrix[2],
                    'R10': rotation_matrix[3], 'R11': rotation_matrix[4], 'R12': rotation_matrix[5],
                    'R20': rotation_matrix[6], 'R21': rotation_matrix[7], 'R22': rotation_matrix[8],
                    'Tx': translation_vector[0], 'Ty': translation_vector[1], 'Tz': translation_vector[2],
                    'FocalLength': focal_length
                })
            except ValueError as e:
                print(f"Ошибка парсинга строки: {line.strip()}")
                print(f"Детали ошибки: {e}")

    return cameras_data

def save_to_csv(cameras_data, output_file):
    # Определение заголовков CSV
    fieldnames = [
        'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22',
        'Tx', 'Ty', 'Tz', 'FocalLength'
    ]

    # Запись данных в CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for data in cameras_data:
            writer.writerow(data)

    print(f"Данные успешно сохранены в {output_file}")

def main():
    # Путь к вашему cameras.txt файлу
    cameras_file = "cameras.txt"
    
    # Путь для сохранения CSV
    output_csv = "camera_coordinates.csv"
    
    # Парсинг файла камер
    cameras_data = parse_cameras(cameras_file)
    
    if cameras_data:
        # Сохранение данных в CSV
        save_to_csv(cameras_data, output_csv)
    else:
        print("Нет данных для записи в CSV.")

if __name__ == "__main__":
    main()
