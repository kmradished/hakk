import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def extract_sand_mask(image_path, num_clusters=5, sand_cluster_idx=2):
    """
    Выделяет песок на изображении с помощью кластеризации K-means
    """
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None
    
    # Конвертируем в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Изменяем форму для кластеризации
    pixels = image_rgb.reshape(-1, 3)
    
    # Применяем K-means кластеризацию
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Получаем метки и центры кластеров
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Создаем маску для песка (обычно песок имеет желтовато-коричневые оттенки)
    # Выбираем кластер с наиболее песчаным цветом (обычно самый яркий желто-коричневый)
    sand_mask = (labels == sand_cluster_idx).reshape(image.shape[:2])
    
    return sand_mask, image_rgb

def highlight_sand(image_path, output_path=None, num_clusters=5, sand_cluster_idx=2):
    """
    Выделяет песок зеленым цветом на изображении
    """
    # Получаем маску песка
    sand_mask, image_rgb = extract_sand_mask(image_path, num_clusters, sand_cluster_idx)
    
    if sand_mask is None:
        return None
    
    # Создаем копию изображения
    highlighted = image_rgb.copy()
    
    # Создаем зеленый цвет для выделения (BGR формат для OpenCV)
    green_color = [0, 255, 0]  # Зеленый в RGB
    
    # Применяем зеленый цвет к областям с песком
    # Используем альфа-смешивание для более естественного вида
    alpha = 0.5  # Прозрачность зеленого слоя
    
    # Создаем зеленый слой
    green_layer = np.zeros_like(highlighted)
    green_layer[sand_mask] = green_color
    
    # Накладываем зеленый слой с прозрачностью
    highlighted = cv2.addWeighted(highlighted, 1, green_layer, alpha, 0)
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Оригинальное изображение
    axes[0].imshow(image_rgb)
    axes[0].set_title('Оригинальное изображение')
    axes[0].axis('off')
    
    # Маска песка
    axes[1].imshow(sand_mask, cmap='gray')
    axes[1].set_title('Маска песка')
    axes[1].axis('off')
    
    # Результат с выделением
    axes[2].imshow(highlighted)
    axes[2].set_title('Песок выделен зеленым')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Сохраняем результат, если указан путь
    if output_path:
        # Конвертируем обратно в BGR для сохранения
        highlighted_bgr = cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, highlighted_bgr)
        print(f"Результат сохранен в: {output_path}")
    
    return highlighted

def process_multiple_images(image_paths, num_clusters=5):
    """
    Обрабатывает несколько изображений
    """
    results = []
    
    for i, img_path in enumerate(image_paths):
        print(f"Обработка изображения {i+1}: {img_path}")
        
        # Пробуем разные индексы кластеров для песка
        for sand_idx in range(num_clusters):
            try:
                result = highlight_sand(
                    img_path, 
                    output_path=f'result_{i}_cluster_{sand_idx}.jpg',
                    num_clusters=num_clusters,
                    sand_cluster_idx=sand_idx
                )
                if result is not None:
                    results.append(result)
                    print(f"  Кластер {sand_idx}: успешно")
                break  # Если удалось обработать, переходим к следующему изображению
            except Exception as e:
                print(f"  Кластер {sand_idx}: ошибка - {e}")
    
    return results

# Обработка ваших изображений
if __name__ == "__main__":
    # Укажите пути к вашим изображениям
    image_paths = ['20.jpg', '25.jpg']
    
    # Основная обработка с параметрами по умолчанию
    process_multiple_images(image_paths, num_clusters=5)
    
    # Дополнительная обработка с разными параметрами для лучшего результата
    print("\nДополнительная обработка с различными параметрами...")
    
    for img_path in image_paths:
        print(f"\nАнализ изображения: {img_path}")
        
        # Пробуем разные параметры
        for n_clusters in [3, 4, 5, 6]:
            print(f"\nКоличество кластеров: {n_clusters}")
            
            # Загружаем изображение для анализа
            image = cv2.imread(img_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Анализ цветов для выбора правильного кластера
                pixels = image_rgb.reshape(-1, 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Выводим цвета центроидов для анализа
                print("Цвета центроидов кластеров (RGB):")
                for j, center in enumerate(kmeans.cluster_centers_):
                    print(f"  Кластер {j}: R={center[0]:.1f}, G={center[1]:.1f}, B={center[2]:.1f}")
                    
                    # Эвристика для определения песка: ищем желто-коричневые оттенки
                    # Песок обычно имеет высокие значения R и G, среднее B
                    if center[0] > 150 and center[1] > 120 and center[2] < 150:
                        print(f"    → Возможный песок (кластер {j})")
                        
                        # Обрабатываем с этим кластером
                        highlight_sand(
                            img_path,
                            output_path=f'{img_path.split(".")[0]}_sand_clusters_{n_clusters}_idx_{j}.jpg',
                            num_clusters=n_clusters,
                            sand_cluster_idx=j
                        )