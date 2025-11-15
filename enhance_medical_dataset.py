# enhance_medical_dataset.py
import json
from pathlib import Path
import re
from collections import Counter

def enhance_medical_dataset():
    """Улучшение медицинского датасета"""
    
    input_dir = Path("cleaned_dataset")
    output_dir = Path("enhanced_dataset")
    output_dir.mkdir(exist_ok=True)
    
    print("УЛУЧШЕНИЕ МЕДИЦИНСКОГО ДАТАСЕТА")
    print("=" * 50)
    
    enhanced_count = 0
    skipped_count = 0
    
    for json_file in input_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Проверяем, есть ли достаточно данных
            if not has_sufficient_data(data):
                print(f"Пропущен (мало данных): {json_file.name}")
                skipped_count += 1
                continue
            
            # Улучшаем структуру данных
            enhanced_data = enhance_data_structure(data)
            
            # Сохраняем улучшенную версию
            output_file = output_dir / json_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
            
            enhanced_count += 1
            if enhanced_count % 50 == 0:
                print(f"Обработано: {enhanced_count} файлов")
                
        except Exception as e:
            print(f"Ошибка в {json_file.name}: {e}")
            skipped_count += 1
    
    print(f"\nРЕЗУЛЬТАТ:")
    print(f"Улучшено файлов: {enhanced_count}")
    print(f"Пропущено файлов: {skipped_count}")
    return enhanced_count

def has_sufficient_data(data):
    """Проверяет, достаточно ли данных в файле"""
    if 'sections' not in data:
        return False
    
    # Считаем общее количество символов во всех разделах
    total_chars = 0
    for section_content in data['sections'].values():
        if section_content and len(str(section_content).strip()) > 0:
            total_chars += len(str(section_content).strip())
    
    # Если меньше 500 символов - считаем недостаточным
    return total_chars >= 500

def enhance_data_structure(data):
    """Улучшает структуру медицинских данных"""
    
    enhanced = {
        'title': data.get('title', '').strip(),
        'sections': {},
        'metadata': {}
    }
    
    # Обрабатываем разделы
    if 'sections' in data:
        for section_name, section_content in data['sections'].items():
            if section_content and len(str(section_content).strip()) > 10:
                cleaned_content = clean_text(section_content)
                enhanced['sections'][section_name] = cleaned_content
    
    # Добавляем метаданные
    enhanced['metadata'] = extract_metadata(enhanced)
    
    return enhanced

def clean_text(text):
    """Очищает и форматирует текст"""
    if not text:
        return ""
    
    text = str(text).strip()
    
    # Удаляем лишние пробелы и переносы
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text

def extract_metadata(data):
    """Извлекает метаданные из медицинского текста"""
    
    all_text = ' '.join([
        data.get('title', ''),
        *[str(content) for content in data.get('sections', {}).values()]
    ]).lower()
    
    # Определяем категории заболеваний
    categories = categorize_disease(all_text)
    
    # Извлекаем симптомы
    symptoms = extract_symptoms(all_text)
    
    # Определяем сложность
    total_chars = sum(len(str(content)) for content in data.get('sections', {}).values())
    complexity = 'высокая' if total_chars > 10000 else 'средняя' if total_chars > 3000 else 'низкая'
    
    # Считаем заполненность разделов
    sections = data.get('sections', {})
    filled_sections = sum(1 for content in sections.values() 
                         if content and len(str(content).strip()) > 50)
    
    return {
        'categories': categories,
        'symptoms': symptoms,
        'symptoms_count': len(symptoms),
        'complexity': complexity,
        'filled_sections': filled_sections,
        'total_sections': len(sections),
        'completeness_score': filled_sections / len(sections) if sections else 0
    }

def categorize_disease(text):
    """Определяет категории заболевания по тексту"""
    
    categories = {
        'инфекционные': ['инфекци', 'вирус', 'бактери', 'заражен', 'эпидеми', 'микроб', 'возбудит'],
        'сердечно-сосудистые': ['сердц', 'артери', 'вен', 'кровяное давление', 'инфаркт', 'инсульт', 'гипертония', 'гипотония'],
        'желудочно-кишечные': ['желудок', 'кишечник', 'пищевод', 'печен', 'желч', 'панкреас', 'аппетит', 'тошнота', 'рвота', 'диарея', 'запор'],
        'дыхательные': ['дыхан', 'легк', 'бронх', 'трахе', 'кашель', 'насморк', 'горло', 'пневмония', 'астма'],
        'неврологические': ['мозг', 'нерв', 'головная боль', 'мигрень', 'паралич', 'судорог', 'памят', 'сознание'],
        'эндокринные': ['гормон', 'щитовидн', 'диабет', 'инсулин', 'обмен веществ'],
        'костно-мышечные': ['кост', 'мышц', 'сустав', 'перелом', 'артрит', 'артроз', 'остеопороз'],
        'мочеполовые': ['почк', 'мочевой', 'половой', 'менструац', 'беременност'],
        'дерматологические': ['кож', 'сып', 'зуд', 'покраснен', 'шелушен'],
        'психические': ['депрессия', 'тревог', 'психоз', 'настроен', 'сон', 'стресс']
    }
    
    detected_categories = []
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            detected_categories.append(category)
    
    return detected_categories if detected_categories else ['другое']

def extract_symptoms(text):
    """Извлекает симптомы из текста"""
    
    symptoms_list = [
        'боль', 'температура', 'тошнота', 'рвота', 'головокружение', 
        'слабость', 'усталость', 'кашель', 'насморк', 'боль в горле',
        'одышка', 'сыпь', 'зуд', 'отек', 'кровотечение', 'диарея', 
        'запор', 'лихорадка', 'озноб', 'потеря аппетита', 'потеря веса',
        'боль в груди', 'боль в животе', 'боль в суставах', 'боль в мышцах',
        'нарушение сна', 'головная боль', 'изжога', 'отрыжка', 'вздутие живота',
        'учащенное сердцебиение', 'повышенное давление', 'пониженное давление',
        'нарушение зрения', 'шум в ушах', 'обморок', 'судороги', 'онемение'
    ]
    
    detected_symptoms = []
    for symptom in symptoms_list:
        if symptom in text:
            detected_symptoms.append(symptom)
    
    return detected_symptoms

if __name__ == "__main__":
    enhanced_count = enhance_medical_dataset()
    
    if enhanced_count > 0:
        print(f"\nСоздана улучшенная версия датасета в папке 'enhanced_dataset'")
        print("Следующий шаг: обновите run_bot.py для использования улучшенного датасета")
    else:
        print("\nНе удалось улучшить датасет. Проверьте исходные данные.")