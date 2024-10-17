import cv2
import numpy as np
import os

def equirectangular_to_cubemap(img, face_size=512):
    """
    Преобразует экварунтную панораму в 4 грани кубической проекции (без top и bottom).
    
    :param img: Входное экварунтное изображение.
    :param face_size: Размер каждой грани куба (по умолчанию 512x512).
    :return: Список из 4 граней куба как изображений.
    """
    # Определение углов для каждой грани куба
    face_mapping = {
        'front': 0,
        'right': 1,
        'back': 2,
        'left': 3
    }
    
    # Список граней в порядке: front, right, back, left
    faces = []
    
    # Ширина и высота панорамы
    height, width, _ = img.shape
    
    # Углы поворота для каждой грани (Yaw, Pitch, Roll)
    # Направления: front (0,0), right (90,0), back (180,0), left (-90,0)
    face_angles = {
        'front': (0, 0, 0),
        'right': (90, 0, 0),
        'back': (180, 0, 0),
        'left': (-90, 0, 0)
    }
    
    for face in ['front', 'right', 'back', 'left']:
        yaw, pitch, roll = face_angles[face]
        face_img = perspective_projection(img, face_size, yaw, pitch, roll)
        faces.append(face_img)
    
    return faces

def perspective_projection(equirect_img, face_size, yaw, pitch, roll):
    """
    Генерирует перспективное изображение для одной грани куба.
    
    :param equirect_img: Входное экварунтное изображение.
    :param face_size: Размер выходного изображения.
    :param yaw: Угол поворота по горизонтали (в градусах).
    :param pitch: Угол поворота по вертикали (в градусах).
    :param roll: Угол поворота вокруг оси взгляда (не используется в данном примере).
    :return: Перспективное изображение для одной грани.
    """
    # Углы обзора (FOV) для кубической проекции обычно 90 градусов
    fov = 90
    # Конвертируем углы в радианы
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)
    
    # Создаем сетку координат для выходного изображения
    x = np.linspace(-1, 1, face_size)
    y = np.linspace(-1, 1, face_size)
    xv, yv = np.meshgrid(x, -y)  # Инвертируем y для корректного отображения
    zv = np.ones_like(xv)
    
    # Нормализуем векторы
    norm = np.sqrt(xv**2 + yv**2 + zv**2)
    xv /= norm
    yv /= norm
    zv /= norm
    
    # Угол для масштабирования (FOV)
    theta = np.deg2rad(fov / 2)
    xv = xv * np.tan(theta)
    yv = yv * np.tan(theta)
    zv = zv * np.ones_like(xv)
    
    # Вращение матрицы (yaw, pitch, roll)
    R_yaw = rotation_matrix(yaw, axis='y')
    R_pitch = rotation_matrix(pitch, axis='x')
    R_roll = rotation_matrix(roll, axis='z')
    R = R_roll @ R_pitch @ R_yaw
    
    # Применяем вращение
    vec = np.stack([xv, yv, zv], axis=-1)  # shape: (face_size, face_size, 3)
    vec = vec.reshape(-1, 3).T  # shape: (3, face_size*face_size)
    vec_rot = R @ vec
    vec_rot = vec_rot.T  # shape: (face_size*face_size, 3)
    
    # Преобразуем обратно в сферические координаты
    lon = np.arctan2(vec_rot[:,0], vec_rot[:,2])
    lat = np.arcsin(vec_rot[:,1])
    
    # Преобразуем координаты в пиксели исходного изображения
    src_x = (lon + np.pi) / (2 * np.pi) * (equirect_img.shape[1] - 1)
    src_y = (np.pi/2 - lat) / np.pi * (equirect_img.shape[0] - 1)
    
    # Интерполяция
    map_x = src_x.reshape(face_size, face_size).astype(np.float32)
    map_y = src_y.reshape(face_size, face_size).astype(np.float32)
    face_img = cv2.remap(equirect_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    return face_img

def rotation_matrix(angle, axis='y'):
    """
    Создает матрицу вращения для заданного угла и оси.
    
    :param angle: Угол вращения в радианах.
    :param axis: Ось вращения ('x', 'y', 'z').
    :return: 3x3 матрица вращения.
    """
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid rotation axis")
    return R

def process_panoramas(input_folder, output_folder, face_size=512):
    """
    Обрабатывает все сферические панорамы в заданной папке и сохраняет кубические грани.
    
    :param input_folder: Путь к папке с панорамными изображениями.
    :param output_folder: Путь к папке для сохранения перспективных изображений.
    :param face_size: Размер каждой грани куба.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Не удалось загрузить изображение: {img_path}")
                continue
            
            print(f"Обрабатывается: {filename}")
            cubemap_faces = equirectangular_to_cubemap(img, face_size)
            
            base_name = os.path.splitext(filename)[0]
            face_names = ['front', 'right', 'back', 'left']
            for face, face_img in zip(face_names, cubemap_faces):
                output_path = os.path.join(output_folder, f"{base_name}_{face}.jpg")
                cv2.imwrite(output_path, face_img)
                print(f"Сохранено: {output_path}")

if __name__ == "__main__":
    # Путь к папке с панорамными изображениями
    input_folder = 'TestTaskSFM\sphere_sfm'
    # Путь к папке для сохранения перспективных изображений
    output_folder = 'persprctive_task2'
    # Размер каждой грани куба
    face_size = 1024  # Можно изменить по необходимости
    
    process_panoramas(input_folder, output_folder, face_size)
