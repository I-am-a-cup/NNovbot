# 🏛️ NN.Travel.AI - AI-помощник туриста по Нижнему Новгороду

Телеграм-бот на базе искусственного интеллекта для создания персональных маршрутов прогулок по Нижнему Новгороду на основе твоих интересов и доступного времени.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ Возможности

🎯 **Персональные маршруты** — создаёт уникальные маршруты на основе твоих интересов  
⏱️ **Умное планирование** — учитывает доступное время (от 30 минут до 12 часов)  
📍 **Учёт локации** — строит маршрут от твоего текущего местоположения  
🕐 **Реальное время** — таймлайн привязан к текущему времени суток  
🔍 **Умный поиск** — гибридный поиск (семантический + ключевые слова)  
💬 **Простой интерфейс** — всего 3 вопроса для создания маршрута  
⚡ **Быстрая работа** — кэширование эмбеддингов для мгновенного ответа  

## 🛠️ Технологии

- **Python 3.10+**
- **pyTelegramBotAPI** — асинхронная работа с Telegram Bot API
- **pandas / numpy** — обработка данных и вычисления
- **scikit-learn** — машинное обучение (TF-IDF, косинусное сходство)
- **aiohttp** — асинхронные HTTP-запросы к OpenRouter API
- **python-dotenv** — управление переменными окружения
- **OpenRouter API** — интеграция LLM моделей (GPT)

## 📋 Требования

- Python 3.10 или выше
- Telegram аккаунт
- Telegram Bot Token (от [@BotFather](https://t.me/BotFather))
- OpenRouter API Key (с сайта [openrouter.ai](https://openrouter.ai/))

## 🚀 Быстрый старт

### 1. Клонируй репозиторий

```bash
git clone https://github.com/yourusername/NNovBot.git
cd NNovBot
```

### 2. Установи зависимости

```bash
pip install -r requirements.txt
```

Или с виртуальным окружением (рекомендуется):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Настрой .env (если нужно)

Файл `.env` уже включён в репозиторий. Если хочешь использовать **свои токены**, отредактируй его:

```bash
nano .env  # или любой редактор
```

Замени значения:

```env
TELEGRAM_BOT_TOKEN=твой_токен_от_BotFather
OPENROUTER_API_KEY=твой_ключ_от_OpenRouter
```

**Как получить токены:**

**🤖 Telegram Bot Token:**
1. Открой Telegram и найди [@BotFather](https://t.me/BotFather)
2. Отправь `/newbot`
3. Следуй инструкциям (название, username)
4. Получишь токен формата: `123456789:ABCdefGHIjklMNOpqrSTUvwxYZ`

**🔑 OpenRouter API Key:**
1. Зайди на [openrouter.ai](https://openrouter.ai/)
2. Зарегистрируйся (бесплатно)
3. Перейди в раздел "Keys"
4. Создай новый ключ
5. Получишь ключ формата: `sk-or-v1-...`

### 4. Запусти бота

```bash
python chatbot.py
```

**При первом запуске:**
- Создаётся кэш эмбеддингов (`embeddings_cultural_sites.npz`)
- Занимает 5-10 секунд
- Последующие запуски мгновенны

Ты увидишь в консоли:
```
INFO: ✓ Excel загружен: 142 объектов
INFO: ✓ Эмбеддинги из кэша (или ↳ Создаю эмбеддинги...)
INFO: 🤖 AI-помощник туриста запущен!
```

### 5. Протестируй бота

1. Открой Telegram
2. Найди своего бота (username который дал BotFather)
3. Отправь `/start`

**Бот задаст 3 вопроса:**

**1️⃣ Что тебя интересует?**  
Примеры: "история", "архитектура", "музеи", "парки", "кофейни", "стрит-арт"

**2️⃣ Сколько времени у тебя есть?**  
Примеры: "2" (часа), "1.5", "3"

**3️⃣ Где ты сейчас?**  
Примеры: "Кремль", "Большая Покровская", "площадь Минина", "ТЦ Небо"

**Результат:**
- 3-5 мест в зависимости от времени
- Объяснение почему каждое место подходит
- Точные адреса
- Время на каждое место
- Таймлайн привязанный к текущему времени

## 📂 Структура проекта

```
NNovBot/
├── chatbot.py                          # Основной код бота (465 строк)
├── cultural_objects_mnn.xlsx           # База данных (142 объекта)
├── .env                                # Переменные окружения с токенами
├── requirements.txt                    # Python-зависимости
└── README.md                           # Этот файл
```

## 📊 База данных культурных объектов

**142 объекта** Нижнего Новгорода в Excel файле `cultural_objects_mnn.xlsx`

### Структура данных:

| Колонка | Описание | Пример |
|---------|----------|--------|
| **id** | Уникальный идентификатор | 57 |
| **title** | Название места | Памятник Петру 1 |
| **address** | Точный адрес | Нижний Новгород, Нижне-Волжская набережная |
| **coordinate** | GPS координаты | POINT (44.003277 56.331576) |
| **description** | HTML описание с историей | `<p>Памятник Петру 1 открыт 24 сентября 2014 года...</p>` |
| **category_id** | Категория | 1 (памятники), 2 (парки), 10 (арт) |
| **url** | Ссылка (необязательно) | https://... |

## ⚙️ Конфигурация

### Настройки в `chatbot.py`:

```python
# База данных
EXCEL_PATH = "cultural_objects_mnn.xlsx"
SHEET_NAME = None  # Автовыбор листа

# OpenRouter API
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_CHAT = "openai/gpt-oss-20b:free"  # Бесплатная модель

# Параметры поиска
TOP_K = 15                # Количество мест для поиска
MIN_SIM = 0.05            # Минимальный порог сходства (0.0-1.0)
CTX_CHAR_BUDGET = 20000   # Лимит символов контекста для LLM
TRIM_CHARS = 4000         # Максимум символов на одно место

# Retry логика
MAX_RETRIES = 5           # Количество повторных попыток при ошибке API
RETRY_DELAY = 30          # Задержка между попытками (секунды)

# Кэш
EMB_CACHE = "embeddings_cultural_sites.npz"
```

### Доступные бесплатные модели OpenRouter:

Можешь заменить `MODEL_CHAT`:

| Модель | Описание |
|--------|----------|
| `openai/gpt-oss-20b:free` | **По умолчанию**, хороший баланс |
| `meta-llama/llama-3.2-3b-instruct:free` | Быстрая, лёгкая |
| `microsoft/phi-3-mini-128k-instruct:free` | Компактная, большой контекст |

Список всех моделей: [openrouter.ai/models](https://openrouter.ai/models)

## 🔍 Как работает алгоритм

### 1. Обработка запроса

```python
User Query: "история архитектура"
    ↓
TF-IDF Vectorization
    ↓
Hash-based embedding (256 dimensions)
    ↓
Normalized vector
```

### 2. Гибридный поиск

```python
Semantic Search (Cosine Similarity)
    ↓ top_k × 2
Результаты по семантике

Keyword Matching (Set Intersection)
    ↓
Результаты по ключевым словам

    ↓
Combined Ranking (с буст-коэффициентом +0.3 для ключевых слов)
    ↓
Фильтрация по MIN_SIM и CTX_CHAR_BUDGET
    ↓
Top K результатов
```

### 3. Генерация маршрута

```python
Context Block (выбранные места)
    ↓
LLM Prompt с инструкциями:
  • Выбери N мест (3-5 в зависимости от времени)
  • Объясни ПОЧЕМУ каждое место подходит
  • Укажи ТОЧНЫЕ адреса
  • Распределить РАЗНОЕ время (30-90 мин)
  • Создать таймлайн с текущим временем
    ↓
OpenRouter API (GPT модель)
    ↓
Post-processing:
  • Удаление таблиц (| и ---)
  • Удаление маркеров (**, ##, •)
  • Форматирование текста
    ↓
Финальный маршрут
```

### 4. UX взаимодействие

```
/start → Вопрос 1
User answer → Edit message → Вопрос 2
User answer → Edit message → Вопрос 3
User answer → Edit message → "Генерирую..."
    ↓ (API call)
Edit message → Готовый маршрут
```

**Фишка:** Бот редактирует одно сообщение вместо отправки новых → чистый интерфейс

## 📊 Производительность

| Метрика | Значение |
|---------|----------|
| **Первый запуск** | ~5-10 сек (генерация эмбеддингов) |
| **Последующие запуски** | Мгновенно (кэш загружен) |
| **Поиск мест** | <1 сек (векторные операции) |
| **Генерация маршрута** | 5-15 сек (зависит от OpenRouter) |
| **Память** | ~50-100 МБ |
| **CPU** | Низкая (ожидание I/O) |

## 🐛 Решение проблем

### ❌ RuntimeError: Положи OPENROUTER_API_KEY и TELEGRAM_BOT_TOKEN в .env

**Причина:** Файл `.env` не найден или пустой

**Решение:**
```bash
# Проверь наличие файла
ls -la .env

# Проверь содержимое
cat .env

# Убедись что токены заполнены (не пустые)
```

### ❌ Бот не отвечает в Telegram

**Проверь:**
1. ✅ Бот запущен (`python chatbot.py`)
2. ✅ В консоли нет ошибок
3. ✅ Токен правильный (проверь в @BotFather)
4. ✅ Интернет работает

### ❌ HTTP 429 (Too Many Requests)

**Причина:** Превышен лимит запросов OpenRouter

**Решение:**
- Бот автоматически повторит запрос (до 5 раз с задержкой)
- Подожди 30-60 секунд
- Рассмотри платный план OpenRouter

### ❌ HTTP 500/502/503 (Server Error)

**Причина:** Проблемы на стороне OpenRouter

**Решение:**
- Бот повторит запрос автоматически
- Проверь статус: [status.openrouter.ai](https://status.openrouter.ai/)
- Попробуй другую модель

### ❌ Плохие результаты поиска

**Настрой параметры:**

```python
# Для большего кол-ва результатов
TOP_K = 20  # было 15

# Для менее строгой фильтрации
MIN_SIM = 0.02  # было 0.05

# Для большего контекста LLM
CTX_CHAR_BUDGET = 30000  # было 20000
```

### ❌ Ошибка с кэшем эмбеддингов

**Решение:**
```bash
# Удали кэш
rm embeddings_cultural_sites.npz

# Перезапусти бота (создастся новый кэш)
python chatbot.py
```

### ❌ Проблемы с Excel файлом

**Проверь:**
```bash
# Файл существует?
ls -la cultural_objects_mnn.xlsx

# Размер нормальный?
du -h cultural_objects_mnn.xlsx
```

Если Excel повреждён, скачай оригинал из репозитория.

## 🚢 Развёртывание на сервере

### Вариант 1: VPS (DigitalOcean, Hetzner, AWS)

```bash
# 1. Подключись к серверу
ssh user@your-server-ip

# 2. Установи зависимости
sudo apt update && sudo apt install -y python3 python3-pip git

# 3. Клонируй репозиторий
git clone https://github.com/yourusername/NNovBot.git
cd NNovBot

# 4. Установи пакеты
pip3 install -r requirements.txt

# 5. (Опционально) Отредактируй .env если нужны другие токены
nano .env

# 6. Запусти в фоне с nohup
nohup python3 chatbot.py > bot.log 2>&1 &

# 7. Проверь логи
tail -f bot.log

# Остановить бота:
pkill -f chatbot.py
```

### Вариант 2: Systemd (Linux)

Создай сервис `/etc/systemd/system/nnovbot.service`:

```ini
[Unit]
Description=NNovBot Telegram Bot
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/NNovBot
Environment="PATH=/home/youruser/NNovBot/venv/bin"
ExecStart=/home/youruser/NNovBot/venv/bin/python chatbot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Запусти:

```bash
sudo systemctl daemon-reload
sudo systemctl enable nnovbot
sudo systemctl start nnovbot
sudo systemctl status nnovbot

# Логи
journalctl -u nnovbot -f
```

### Вариант 3: Docker

```bash
# Собери образ
docker build -t nnovbot .

# Запусти контейнер
docker run -d --name nnovbot --restart unless-stopped --env-file .env nnovbot

# Логи
docker logs -f nnovbot

# Остановить
docker stop nnovbot
```

### Вариант 4: Screen (простой способ)

```bash
# Установи screen
sudo apt install screen

# Создай сессию
screen -S nnovbot

# Запусти бота
python3 chatbot.py

# Отключись от сессии (бот продолжит работу)
# Нажми: Ctrl+A, затем D

# Вернуться к сессии
screen -r nnovbot

# Завершить сессию
screen -X -S nnovbot quit
```

## 📝 Лицензия

MIT License — можешь использовать в коммерческих и личных проектах.

См. файл [LICENSE](LICENSE) для деталей.
