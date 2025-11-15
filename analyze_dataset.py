# analyze_dataset.py - исправленная версия без emoji
import json
from pathlib import Path
from collections import Counter
import re

def analyze_dataset(data_dir):
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob("*.json"))
    
    print(f"АНАЛИЗ ДАТАСЕТА: {len(json_files)} файлов...")
    
    # Сбор статистики
    total_diseases = 0
    sections_counter = Counter()
    text_lengths = []
    missing_sections = 0
    file_errors = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_diseases += 1
            title = data.get('title', 'Без названия')
            
            # Анализ разделов
            if 'sections' in data:
                for section in data['sections']:
                    sections_counter[section] += 1
                    text = data['sections'][section]
                    if not text or len(str(text).strip()) < 10:
                        missing_sections += 1
                    else:
                        text_lengths.append(len(str(text)))
            else:
                missing_sections += 1
                
        except Exception as e:
            print(f"Ошибка в файле {json_file.name}: {e}")
            file_errors += 1
    
    print(f"\nСТАТИСТИКА ДАТАСЕТА:")
    print(f"• Всего заболеваний: {total_diseases}")
    print(f"• Файлов с ошибками: {file_errors}")
    print(f"• Распределение разделов:")
    for section, count in sections_counter.most_common():
        print(f"  - {section}: {count}")
    
    if text_lengths:
        avg_length = sum(text_lengths) / len(text_lengths)
        print(f"• Средняя длина текста: {avg_length:.0f} символов")
        print(f"• Минимальная длина: {min(text_lengths)} символов")
        print(f"• Максимальная длина: {max(text_lengths)} символов")
    
    print(f"• Пропущенных разделов: {missing_sections}")
    
    # Анализ качества данных
    print(f"\nКАЧЕСТВО ДАННЫХ:")
    if total_diseases > 0:
        coverage = (total_diseases - file_errors) / total_diseases * 100
        print(f"• Покрытие данных: {coverage:.1f}%")
    
    if sections_counter:
        most_common_section = sections_counter.most_common(1)[0]
        least_common_section = sections_counter.most_common()[-1]
        print(f"• Самый частый раздел: '{most_common_section[0]}' ({most_common_section[1]} раз)")
        print(f"• Самый редкий раздел: '{least_common_section[0]}' ({least_common_section[1]} раз)")
    
    return {
        'total_diseases': total_diseases,
        'sections_distribution': dict(sections_counter),
        'avg_text_length': avg_length if text_lengths else 0,
        'file_errors': file_errors
    }

def check_data_quality(data_dir):
    """Проверка конкретных проблем в данных"""
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob("*.json"))
    
    print(f"\nДЕТАЛЬНЫЙ АНАЛИЗ ПРОБЛЕМ:")
    
    problems = {
        'no_title': [],
        'empty_sections': [],
        'short_texts': [],
        'no_sections': []
    }
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Проверка заголовка
            title = data.get('title', '').strip()
            if not title:
                problems['no_title'].append(json_file.name)
            
            # Проверка разделов
            if 'sections' not in data:
                problems['no_sections'].append(json_file.name)
            else:
                for section_name, section_content in data['sections'].items():
                    if not section_content or len(str(section_content).strip()) < 20:
                        problems['empty_sections'].append(f"{json_file.name}: {section_name}")
                    
                    if section_content and len(str(section_content).strip()) < 100:
                        problems['short_texts'].append(f"{json_file.name}: {section_name}")
                        
        except Exception as e:
            continue
    
    # Вывод проблем
    for problem_type, files in problems.items():
        if files:
            print(f"\nПРОБЛЕМА: {problem_type}")
            for file in files[:5]:  # Показываем первые 5 примеров
                print(f"  - {file}")
            if len(files) > 5:
                print(f"  ... и еще {len(files) - 5} файлов")
    
    return problems

if __name__ == "__main__":
    print("=" * 60)
    print("АНАЛИЗ МЕДИЦИНСКОГО ДАТАСЕТА")
    print("=" * 60)
    
    stats = analyze_dataset("cleaned_dataset")
    problems = check_data_quality("cleaned_dataset")
    
    print(f"\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ:")
    
    if stats['total_diseases'] == 0:
        print("1. Добавьте JSON-файлы в папку 'cleaned_dataset'")
        print("2. Убедитесь, что файлы имеют правильный формат")
    elif stats['file_errors'] > 0:
        print("1. Исправьте ошибки в JSON-файлах")
        print("2. Проверьте кодировку файлов (должна быть UTF-8)")
    elif problems['no_title']:
        print("1. Добавьте заголовки (title) во все файлы")
    elif problems['empty_sections']:
        print("1. Заполните пустые разделы данными")
    else:
        print("1. Датасет в хорошем состоянии! Можно переходить к улучшениям")
        print("2. Рекомендуется создать обогащенную версию датасета")
    
    print("=" * 60)