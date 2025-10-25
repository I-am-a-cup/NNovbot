import os
import logging
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import telebot
from dotenv import load_dotenv
import requests
import json
import re

# -------- Настройки --------
EXCEL_PATH = "cultural_objects_mnn.xlsx"
SHEET_NAME = None

OPENROUTER_API_KEY = None
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_CHAT = "openai/gpt-oss-20b:free"

TOP_K      = 10
MIN_SIM    = 0.05
CTX_CHAR_BUDGET = 15000

MAX_RETRIES   = 5
RETRY_DELAY   = 30     
TRIM_CHARS    = 4000  

EMB_CACHE = "embeddings_cultural_sites.npz"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TG_TOKEN           = os.getenv("TELEGRAM_BOT_TOKEN")
if not OPENROUTER_API_KEY or not TG_TOKEN:
    raise RuntimeError("Положи OPENROUTER_API_KEY и TELEGRAM_BOT_TOKEN в .env")

bot = telebot.TeleBot(TG_TOKEN)

# -------- Загрузка БД --------
def load_db():
    path = EXCEL_PATH if os.path.exists(EXCEL_PATH) else "cultural_objects_mnn.xlsx"
    xls = pd.ExcelFile(path)
    if SHEET_NAME:
        sheet = SHEET_NAME
    else:
        sheet = "cultural_sites_202509191434" if "cultural_sites_202509191434" in xls.sheet_names else xls.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet)

    for c in ["title","description","address","coordinate","url"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": "", "None": ""})

    if "category_id" in df.columns:
        df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").fillna(-1).astype(int)

    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df)+1, dtype=int)

    def row_text(r: pd.Series) -> str:
        get = r.get
        title = get('title','')
        addr = get('address','')
        desc = get('description','')
        cat = get('category_id','')
        coord = get('coordinate','')
        
        return (
            f"Название: {title}\n"
            f"Адрес: {addr}\n"
            f"Категория: {cat}\n"
            f"Координаты: {coord}\n"
            f"Описание: {desc}"
        )

    df["__blob__"] = df.apply(row_text, axis=1).astype(str).str.slice(0, TRIM_CHARS)
    logging.info(f"✓ Excel загружен: {len(df)} объектов")
    return df

DB = load_db()

# -------- Эмбеддинги --------
def tfidf_embed(text: str) -> np.ndarray:
    """TF-IDF вектор"""
    words = text.lower().split()
    words = [''.join(c for c in w if c.isalnum() or c in 'ёа') for w in words]
    words = [w for w in words if len(w) > 2]
    
    vec = np.zeros(256, dtype=np.float32)
    
    for word in words:
        h = hash(word) % 256
        vec[h] += 1.0
    
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    
    return vec

def build_or_load_embeddings(df: pd.DataFrame):
    if os.path.exists(EMB_CACHE):
        try:
            data = np.load(EMB_CACHE, allow_pickle=True)
            if "ids" in data and "vecs" in data and "blobs" in data and "titles" in data:
                logging.info("✓ Эмбеддинги из кэша")
                return data["ids"], data["vecs"], data["blobs"], data["titles"]
        except Exception as e:
            logging.warning(f"Ошибка кэша: {e} — пересоздаю")
            try:
                os.remove(EMB_CACHE)
            except:
                pass

    texts = df["__blob__"].tolist()
    titles = df["title"].tolist()
    ids   = df["id"].astype(int).to_numpy()
    
    logging.info("↳ Создаю эмбеддинги...")
    vecs = np.array([tfidf_embed(t) for t in texts], dtype=np.float32)
    blobs = np.array(texts, dtype=object)
    titles_arr = np.array(titles, dtype=object)
    
    np.savez_compressed(EMB_CACHE, ids=ids, vecs=vecs, blobs=blobs, titles=titles_arr)
    logging.info("✓ Эмбеддинги сохранены")
    return ids, vecs, blobs, titles_arr

IDS, VECS, BLOBS, TITLES = build_or_load_embeddings(DB)

# -------- Умная проверка релевантности --------
def is_relevant_query(query: str) -> bool:
    """
    Многоуровневая проверка релевантности запроса.
    Возвращает True если запрос про Нижний Новгород и культурные объекты.
    """
    query_lower = query.lower()
    score = 0  # Счётчик уверенности
    
    # 1. Проверка на SQL-инъекции и программирование (жёсткий блок)
    sql_patterns = [
        r'select\s+.*from', r'insert\s+into', r'update\s+.*set',
        r'delete\s+from', r'drop\s+table', r'create\s+table',
        r'alter\s+table', r';\s*--', r'union\s+select'
    ]
    for pattern in sql_patterns:
        if re.search(pattern, query_lower):
            logging.warning(f"⚠️ SQL-инъекция: {query[:50]}")
            return False
    
    # Программирование
    if any(word in query_lower for word in ['python', 'код', 'функция', 'класс', 'import', 'def ']):
        logging.warning(f"⚠️ Программирование: {query[:50]}")
        return False
    
    # 2. Мусорные запросы
    if len(query.strip()) < 3:
        return False
    
    if any(junk in query_lower for junk in ['шкибиди', 'трулялю', 'бегать диван']):
        logging.warning(f"⚠️ Мусор: {query[:50]}")
        return False
    
    # Мат
    if any(bad in query_lower for bad in ['бля', 'ахуел', 'заебался']):
        logging.warning(f"⚠️ Нецензурная лексика")
        return False
    
    # 3. Проверка географической привязки к Нижнему Новгороду (+3 балла)
    nn_places = [
        'нижн', 'новгород', 'кремл', 'волг', 'заречь', 'канавин',
        'автозавод', 'сормов', 'московск', 'ленинск', 'приокск',
        'небо', 'фантастик', 'жар', 'мега', 'рио',  # ТЦ
        'покровск', 'большая покровская', 'площадь', 'минина'
    ]
    for place in nn_places:
        if place in query_lower:
            score += 3
            break
    
    # 4. Проверка на культурные объекты (+2 балла)
    culture_words = [
        'памятник', 'музей', 'храм', 'церковь', 'монастырь', 'собор',
        'мозаик', 'скульптур', 'фреск', 'архитектур', 'исторически',
        'достопримечательност', 'театр', 'галере', 'выставк'
    ]
    for word in culture_words:
        if word in query_lower:
            score += 2
            break
    
    # 5. Проверка на туристические запросы (+2 балла)
    tourist_words = [
        'куда', 'где', 'сходить', 'пойти', 'посетить', 'посмотреть',
        'прогулк', 'маршрут', 'экскурси', 'интересн', 'красив',
        'рекомендуе', 'расскажи', 'информаци', 'описани'
    ]
    for word in tourist_words:
        if word in query_lower:
            score += 2
            break
    
    # 6. Проверка на места общепита и развлечений (+1 балл)
    places_words = ['кафе', 'ресторан', 'кофейн', 'бар', 'парк', 'сквер', 'бульвар']
    for word in places_words:
        if word in query_lower:
            score += 1
            break
    
    # 7. Штрафы за нерелевантные темы
    offtopic_themes = {
        'наука': ['квантов', 'физик', 'химия', 'биология', 'математик'],
        'абстрактные вопросы': ['почему', 'зачем', 'как работает', 'объясни', 'что такое'],
        'другие темы': ['небо голубое', 'трава зелёная', 'вода мокрая']
    }
    
    for theme, words in offtopic_themes.items():
        for word in words:
            if word in query_lower and score < 3:  # Штрафуем только если мало баллов
                score -= 2
                logging.info(f"📉 Штраф за '{theme}': {word}")
                break
    
    # 8. Решение: нужно >= 3 баллов для релевантности
    is_relevant = score >= 3
    
    if not is_relevant:
        logging.info(f"❌ Офф-топик (score={score}): {query[:50]}")
    else:
        logging.info(f"✅ Релевантно (score={score})")
    
    return is_relevant

# -------- Поиск мест --------
def search_top_k(q: str, top_k=TOP_K, min_sim=MIN_SIM):
    q_emb = tfidf_embed(q)
    sims = cosine_similarity([q_emb], VECS)[0]
    semantic_idx = np.argsort(-sims)[:top_k * 2]
    
    q_lower = q.lower()
    keyword_matches = []
    for i, title in enumerate(TITLES):
        title_lower = title.lower()
        q_words = set(q_lower.split())
        title_words = set(title_lower.split())
        if q_words & title_words:
            keyword_matches.append((i, len(q_words & title_words)))
    
    results_dict = {}
    
    for rank, i in enumerate(semantic_idx):
        if sims[i] >= min_sim:
            results_dict[i] = (sims[i], rank, 0)
    
    for rank, (i, matches) in enumerate(sorted(keyword_matches, key=lambda x: -x[1])):
        if i in results_dict:
            sim, sem_rank, _ = results_dict[i]
            results_dict[i] = (sim + 0.3, sem_rank, rank)
        else:
            results_dict[i] = (min(sims[i] + 0.3, 1.0), 999, rank)
    
    sorted_results = sorted(
        results_dict.items(),
        key=lambda x: (-x[1][2] < 0, x[1][2], -x[1][0])
    )
    
    context, total = [], 0
    for i, (sim, _, _) in sorted_results[:top_k * 3]:
        blob = BLOBS[i]
        if total + len(blob) > CTX_CHAR_BUDGET: 
            break
        context.append((float(sim), int(IDS[i]), blob, TITLES[i]))
        total += len(blob)

    return context[:top_k]

def make_context_block(hits):
    """Контекст для GPT"""
    blocks = []
    for i, (sim, _id, blob, title) in enumerate(hits, 1):
        blocks.append(f"{i}. {title}\n{blob}\n")
    return "\n".join(blocks)

# -------- API запрос с умной обработкой ошибок --------
def _make_request(query: str, context: str = "", timeout=120):
    """Запрос к OpenRouter с экспоненциальным бэкофф"""
    if context:
        user_msg = f"На основе информации ниже, ответь на вопрос кратко (2-3 предложения). Не используй таблицы, маркеры, форматирование.\n\n{context}\n\nВопрос: {query}"
    else:
        user_msg = query
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL_CHAT,
        "messages": [
            {
                "role": "system",
                "content": "Ты помощник туриста по Нижнему Новгороду в Telegram. Отвечаешь кратко и по делу. Без таблиц, маркеров и форматирования."
            },
            {
                "role": "user",
                "content": user_msg
            }
        ],
        "temperature": 0.4,
        "top_p": 0.9,
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            logging.warning(f"HTTP {status_code} (попытка {attempt + 1}/{MAX_RETRIES})")
            
            if status_code == 429:  # Too Many Requests
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    logging.warning(f"Лимит запросов. Жду {wait_time} сек...")
                    time.sleep(wait_time)
                    continue
            elif status_code in [500, 502, 503, 504]:  # Server errors
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logging.warning(f"Ошибка сервера {status_code}. Жду {wait_time} сек...")
                    time.sleep(wait_time)
                    continue
            
            raise
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                logging.warning(f"Ошибка сети. Жду {wait_time} сек...")
                time.sleep(wait_time)
                continue
            raise

# -------- Очистка текста --------
def clean_telegram_text(text: str) -> str:
    """Очищает текст для TG"""
    text = re.sub(r'\|.*?\|', '', text)
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'^[\•\-\*]\s+', '— ', text, flags=re.MULTILINE)
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

# -------- GPT запрос --------
def ask_gpt_text(user_q: str, context_block: str = "") -> str:
    try:
        result = _make_request(user_q, context_block)
        
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"].strip()
            answer = clean_telegram_text(answer)
            answer = re.sub(r'[\U0001F300-\U0001F9FF]', '', answer)
            return answer
        else:
            raise Exception(f"Invalid response")
            
    except Exception as e:
        logging.error(f"Ошибка при запросе: {e}")
        raise

# -------- Telegram --------
@bot.message_handler(commands=['start','help'])
def start(msg):
    bot.reply_to(msg,
        "Привет! Я помогу найти интересные места в Нижнем Новгороде.\n\n"
        "Просто напиши, что тебя интересует:\n"
        "— Памятники\n"
        "— Музеи\n"
        "— Храмы\n"
        "— Где погулять\n"
        "— Что посмотреть\n\n"
        "И я подскажу лучшие места!"
    )

@bot.message_handler(content_types=['text'])
def handle(msg):
    try:
        q = msg.text.strip()
        
        if not q or len(q) < 2:
            return
        
        logging.info(f"Q: {q}")
        
        # Проверяем релевантность
        if not is_relevant_query(q):
            bot.send_message(msg.chat.id, "Я помогаю искать интересные места в Нижнем Новгороде. Спроси что-нибудь о памятниках, музеях, храмах или местах для прогулок!")
            return
        
        bot.send_chat_action(msg.chat.id, "typing")
        
        # Ищем места
        hits = search_top_k(q, top_k=TOP_K)
        
        if not hits:
            bot.send_message(msg.chat.id, "Не нашёл подходящих мест по этому запросу.")
            return
        
        logging.info(f"Found {len(hits)} results")
        ctx = make_context_block(hits)
        
        try:
            answer = ask_gpt_text(q, ctx)
            bot.send_message(msg.chat.id, answer)
            logging.info(f"✓ Ответ пользователю: {answer}...")
        except Exception as e:
            logging.error(f"Ошибка при получении ответа: {e}")
            bot.send_message(msg.chat.id, "Сервис перегружен, попробуй позже.")
        
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        bot.send_message(msg.chat.id, "Ошибка. Попробуй ещё раз позже")

# -------- Main --------
if __name__ == "__main__":
    logging.info("🤖 Бот запущен!")
    try:
        bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)
    except KeyboardInterrupt:
        logging.info("Бот остановлен")