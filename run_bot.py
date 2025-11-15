from pathlib import Path
import json
import os
import faiss
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

class MedicalInterviewBot:
    def __init__(self, rebuild_db: bool = False):
        self.script_dir = Path(__file__).parent
        self.data_dir = self.script_dir / "enhanced_dataset"
        
        # СОХРАНЯЕМ В ПРОЕКТЕ вместо temp папки
        self.db_dir = self.script_dir / "faiss_db"
        self.db_dir.mkdir(exist_ok=True)  # Создаем папку если не существует
        
        self.conversation_history = []
        self.collected_info = {
            "chief_complaint": "",
            "symptoms": [],
            "duration": "",
            "additional_info": []
        }
        
        print("=" * 70)
        print("МЕДИЦИНСКИЙ ИНТЕРВЬЮЕР v2.5 (Project DB)")
        print("=" * 70)
        print(f"\nБаза данных: {self.db_dir}")
        
        if not self.data_dir.exists():
            print(f"\nОШИБКА: Папка {self.data_dir} не существует!")
            exit(1)
        
        if rebuild_db and self.db_dir.exists():
            print("\nУдаление старого индекса...")
            import shutil
            shutil.rmtree(self.db_dir)
            # Пересоздаем папку после удаления
            self.db_dir.mkdir(exist_ok=True)
            print("   Удален")
        
        self._load_or_create_knowledge_base()
        
        print("\nИнициализация языковой модели...")
        self.llm = ChatOllama(model="llama3.1", temperature=0.3)
        print("   llama3.1 готова")
        
        print("\n" + "=" * 70)
        print("СИСТЕМА ГОТОВА!")
        print("=" * 70)
    
    def _load_or_create_knowledge_base(self):
        """Загрузка или создание FAISS индекса"""
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Проверяем существует ли индекс
        index_path = self.db_dir / "index.faiss"
        if self.db_dir.exists() and index_path.exists():
            print("\nНайден существующий FAISS индекс")
            print(f"   Путь: {self.db_dir}")
            
            try:
                print("   Загрузка...")
                # Используем load_local вместо pickle
                self.vectorstore = FAISS.load_local(
                    str(self.db_dir),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Проверка
                test = self.vectorstore.similarity_search("тест", k=1)
                print("   Индекс загружен успешно")
                return
                
            except Exception as e:
                print(f"   Ошибка: {e}")
                print("   Создаем новый...")
        
        print("\nСоздание нового FAISS индекса")
        print("   Займет 5-15 минут\n")
        
        self._create_new_database(embeddings)
    
    def _create_new_database(self, embeddings):
        """Создание FAISS индекса"""
        
        # 1. Загрузка документов
        print("1. Загрузка документов...")
        documents = []
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            print("   Нет JSON файлов!")
            exit(1)
        
        total = len(json_files)
        print(f"   Найдено: {total} файлов")
        
        for i, json_file in enumerate(json_files, 1):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                title = data.get("title", "")
                full_text = f"# {title}\n\n"
                
                if "sections" in data:
                    for section_name, section_text in data["sections"].items():
                        if section_text and str(section_text).strip():
                            readable_name = section_name.replace("_", " ").title()
                            full_text += f"## {readable_name}\n{section_text}\n\n"
                
                if len(full_text) > 100:
                    doc = Document(
                        page_content=full_text,
                        metadata={"title": title, "disease": title}
                    )
                    documents.append(doc)
                
                # Прогресс каждые 50 файлов
                if i % 50 == 0 or i == total:
                    print(f"   Обработано: {i}/{total}")
                    
            except Exception as e:
                print(f"   Ошибка в файле {json_file.name}: {e}")
        
        print(f"   Загружено: {len(documents)} заболеваний")
        
        # 2. Разбивка
        print("\n2. Разбивка текста...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        total_splits = len(splits)
        print(f"   Фрагментов: {total_splits}")
        
        # 3. Создание эмбеддингов порциями
        print("\n3. Создание векторных эмбеддингов...")
        print("   Это займет время - наберитесь терпения\n")
        
        batch_size = 100
        vectorstore = None
        
        try:
            for i in range(0, total_splits, batch_size):
                batch = splits[i:i+batch_size]
                
                if vectorstore is None:
                    # Первая порция
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    # Добавляем к существующей
                    vectorstore.add_documents(batch)
                
                # Прогресс
                progress = min(i + batch_size, total_splits)
                percentage = (progress / total_splits) * 100
                print(f"   Прогресс: {progress}/{total_splits} ({percentage:.1f}%)")
            
            self.vectorstore = vectorstore
            
            # 4. Сохранение через save_local
            print("\n4. Сохранение индекса...")
            
            # Используем save_local вместо pickle - это правильный способ
            self.vectorstore.save_local(str(self.db_dir))
            
            print(f"   Сохранено в: {self.db_dir}")
            
        except Exception as e:
            print(f"\n   ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _search_context(self, query: str, k: int = 3) -> str:
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content[:700] for doc in docs])
            return context
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return ""

    def search_by_symptoms(self, symptoms_list):
        """Поиск заболеваний по списку симптомов"""
        try:
            # Загружаем индекс симптомов
            index_path = self.script_dir / "symptoms_index.json"
            if not index_path.exists():
                print("Индекс симптомов не найден. Сначала создайте его.")
                return []
            
            with open(index_path, 'r', encoding='utf-8') as f:
                symptoms_index = json.load(f)
            
            # Ищем совпадения
            matching_diseases = {}
            for symptom in symptoms_list:
                symptom_lower = symptom.lower().strip()
                if symptom_lower in symptoms_index:
                    for disease in symptoms_index[symptom_lower]:
                        title = disease['title']
                        if title not in matching_diseases:
                            matching_diseases[title] = 0
                        matching_diseases[title] += 1
            
            # Сортируем по количеству совпадений симптомов
            sorted_diseases = sorted(matching_diseases.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            return sorted_diseases[:5]  # Топ-5 результатов
            
        except Exception as e:
            print(f"Ошибка поиска по симптомам: {e}")
            return []
    
    def _generate_question(self) -> str:
        """Генерация вопроса с обработкой ошибок"""
        search_query = f"{self.collected_info['chief_complaint']} {' '.join(self.collected_info['symptoms'])}"
        context = self._search_context(search_query, k=2)
        
        # Поиск по симптомам для улучшения контекста
        if self.collected_info['symptoms']:
            symptom_matches = self.search_by_symptoms(self.collected_info['symptoms'])
            if symptom_matches:
                symptom_context = "\nВозможные заболевания по симптомам:\n"
                for disease, count in symptom_matches:
                    symptom_context += f"- {disease} (совпадений: {count})\n"
                context += symptom_context
        
        history = "\n".join([
            f"{'Врач' if m['role'] == 'assistant' else 'Пациент'}: {m['content']}"
            for m in self.conversation_history[-4:]
        ])
    
        prompt = ChatPromptTemplate.from_template("""
Ты врач, собирающий анамнез.

ИСТОРИЯ:
{history}

ИНФОРМАЦИЯ:
- Жалоба: {chief_complaint}
- Симптомы: {symptoms}

КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ:
{context}

Задай ОДИН короткий вопрос для уточнения симптомов.

Вопрос:""")
    
        try:
            # Увеличиваем timeout и добавляем параметры
            from langchain_core.runnables import RunnableConfig
            
            response = self.llm.invoke(
                prompt.format(
                    history=history,
                    chief_complaint=self.collected_info["chief_complaint"] or "не указано",
                    symptoms=", ".join(self.collected_info["symptoms"]) if self.collected_info["symptoms"] else "нет",
                    context=context or "Нет данных"
                ),
                config=RunnableConfig(
                    max_concurrency=1,
                    timeout=30  # 30 секунд timeout
                )
            )
            return response.content.strip()
        except Exception as e:
            print(f"\nОшибка LLM: {e}")
            # Fallback вопросы
            fallback_questions = [
                "Как давно у вас эти симптомы?",
                "Усиливаются ли симптомы после еды?",
                "Есть ли температура?",
                "Была ли рвота?",
                "Где именно локализуется боль?"
            ]
            import random
            return random.choice(fallback_questions)
    
    def _extract_info(self, text: str):
        text_lower = text.lower()
        
        time_words = ['день', 'дня', 'дней', 'неделю', 'месяц', 'год']
        if any(w in text_lower for w in time_words) and not self.collected_info["duration"]:
            self.collected_info["duration"] = text
        
        # Расширенный список симптомов
        symptoms = [
            'боль', 'температура', 'тошнота', 'рвота', 'слабость', 'усталость',
            'кашель', 'насморк', 'горло', 'голова', 'живот', 'головокружение',
            'одышка', 'сыпь', 'зуд', 'отек', 'кровотечение', 'диарея', 'запор',
            'лихорадка', 'озноб', 'потеря аппетита', 'потеря веса', 'боль в груди',
            'боль в суставах', 'боль в мышцах', 'нарушение сна', 'изжога', 'отрыжка',
            'вздутие живота', 'учащенное сердцебиение', 'повышенное давление',
            'пониженное давление', 'нарушение зрения', 'шум в ушах', 'обморок',
            'судороги', 'онемение'
        ]
        
        for symptom in symptoms:
            if symptom in text_lower:
                if symptom not in " ".join(self.collected_info["symptoms"]).lower():
                    self.collected_info["symptoms"].append(symptom)
    
    def _should_continue(self) -> bool:
        questions = len([m for m in self.conversation_history if m["role"] == "assistant"])
        has_info = (
            bool(self.collected_info["chief_complaint"]) and
            (len(self.collected_info["symptoms"]) >= 2 or bool(self.collected_info["duration"]))
        )
        return questions < 8 and not has_info
    
    def _generate_report(self) -> str:
        """Генерация отчёта с обработкой ошибок"""
        search_query = " ".join([
            self.collected_info["chief_complaint"],
            *self.collected_info["symptoms"]
        ])
        context = self._search_context(search_query, k=5)
        
        # Добавляем результаты поиска по симптомам в отчет
        if self.collected_info['symptoms']:
            symptom_matches = self.search_by_symptoms(self.collected_info['symptoms'])
            if symptom_matches:
                symptom_info = "\nСООТВЕТСТВИЕ СИМПТОМАМ:\n"
                for disease, count in symptom_matches:
                    symptom_info += f"- {disease} (совпадение по {count} симптомам)\n"
                context += symptom_info
        
        conversation = "\n".join([
            f"{'Врач' if m['role'] == 'assistant' else 'Пациент'}: {m['content']}"
            for m in self.conversation_history
        ])
        
        prompt = ChatPromptTemplate.from_template("""
Составь медицинский отчёт для врача.

БЕСЕДА:
{conversation}

КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ:
{context}

Формат:

**Anamnesis morbi:**
[История заболевания]

**Differential diagnosis:**
[Возможные диагнозы]

**Recommendations:**
[План обследования]

Отчёт:""")
    
        try:
            from langchain_core.runnables import RunnableConfig
            
            print("   Генерация отчёта (может занять 10-30 секунд)...")
            
            response = self.llm.invoke(
                prompt.format(
                    conversation=conversation,
                    context=context or "Требуется обследование"
                ),
                config=RunnableConfig(
                    timeout=60  # 60 секунд для отчёта
                )
            )
            return response.content
        except Exception as e:
            print(f"\nОшибка генерации: {e}")
            
            # Fallback - простой отчёт
            return f"""**Anamnesis morbi:**
Пациент обратился с жалобами: {self.collected_info['chief_complaint']}
Симптомы: {', '.join(self.collected_info['symptoms']) if self.collected_info['symptoms'] else 'не указаны'}
Длительность: {self.collected_info['duration'] if self.collected_info['duration'] else 'не указана'}

**Differential diagnosis:**
Требуется дополнительное обследование для постановки диагноза.

**Recommendations:**
- Консультация врача
- Общий анализ крови
- УЗИ органов брюшной полости
- При необходимости - дополнительные исследования"""
    
    def start_interview(self):
        print("\n" + "=" * 70)
        print("МЕДИЦИНСКОЕ ИНТЕРВЬЮ")
        print("=" * 70)
        print("\nКоманды: 'стоп' - завершить, 'exit' - выход\n")
        
        greeting = "Здравствуйте! Что вас беспокоит?"
        print(f"Бот: {greeting}\n")
        self.conversation_history.append({"role": "assistant", "content": greeting})
        
        complaint = input("Пациент: ").strip()
        
        if complaint.lower() in ['exit', 'выход']:
            print("\nДо свидания!")
            return
        
        if not complaint:
            print("Введите жалобу")
            return
        
        self.collected_info["chief_complaint"] = complaint
        self.conversation_history.append({"role": "user", "content": complaint})
        self._extract_info(complaint)
        
        while self._should_continue():
            try:
                question = self._generate_question()
                print(f"\nБот: {question}\n")
                self.conversation_history.append({"role": "assistant", "content": question})
                
                answer = input("Пациент: ").strip()
                
                if answer.lower() in ['exit', 'выход']:
                    print("\nДо свидания!")
                    return
                
                if answer.lower() == 'стоп':
                    break
                
                if not answer:
                    continue
                
                self.conversation_history.append({"role": "user", "content": answer})
                self._extract_info(answer)
                
            except Exception as e:
                print(f"Ошибка: {e}")
                break
        
        print("\n" + "=" * 70)
        print("ГЕНЕРАЦИЯ ОТЧЁТА...")
        print("=" * 70)
        
        try:
            report = self._generate_report()
            
            print("\n" + "=" * 70)
            print("МЕДИЦИНСКИЙ ОТЧЁТ")
            print("=" * 70 + "\n")
            print(report)
            print("\n" + "=" * 70)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.script_dir / f"report_{timestamp}.txt"
            
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(f"МЕДИЦИНСКИЙ ОТЧЁТ\n")
                f.write(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
                f.write("=" * 70 + "\n\n")
                f.write(report)
            
            print(f"\nСохранено: {report_file.name}")
            
        except Exception as e:
            print(f"ОШИБКА: {e}")

if __name__ == "__main__":
    import sys
    rebuild = "--rebuild" in sys.argv
    
    try:
        bot = MedicalInterviewBot(rebuild_db=rebuild)
        bot.start_interview()
    except KeyboardInterrupt:
        print("\n\nПрервано")
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()