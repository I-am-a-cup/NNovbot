import os
import logging
import asyncio
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from telebot.async_telebot import AsyncTeleBot
from dotenv import load_dotenv
import aiohttp
import re
from datetime import datetime
from typing import Dict, List, Tuple

# -------- Настройки --------
EXCEL_PATH = "cultural_objects_mnn.xlsx"
SHEET_NAME = None

OPENROUTER_API_KEY = None
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_CHAT = "openai/gpt-oss-20b:free"

TOP_K = 15
MIN_SIM = 0.05
CTX_CHAR_BUDGET = 20000

MAX_RETRIES = 5
RETRY_DELAY = 30
TRIM_CHARS = 4000

EMB_CACHE = "embeddings_cultural_sites.npz"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not OPENROUTER_API_KEY or not TG_TOKEN:
    raise RuntimeError("Положи OPENROUTER_API_KEY и TELEGRAM_BOT_TOKEN в .env")

bot = AsyncTeleBot(TG_TOKEN)
user_states: Dict[int, Dict] = {}

# -------- Загрузка БД --------
def load_db():
    path = EXCEL_PATH if os.path.exists(EXCEL_PATH) else "cultural_objects_mnn.xlsx"
    xls = pd.ExcelFile(path)
    if SHEET_NAME:
        sheet = SHEET_NAME
    else:
        sheet = "cultural_sites_202509191434" if "cultural_sites_202509191434" in xls.sheet_names else xls.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet)

    for c in ["title", "description", "address", "coordinate", "url"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": "", "None": ""})

    if "category_id" in df.columns:
        df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").fillna(-1).astype(int)

    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1, dtype=int)

    def row_text(r: pd.Series) -> str:
        get = r.get
        return (
            f"Название: {get('title', '')}\n"
            f"Адрес: {get('address', '')}\n"
            f"Категория: {get('category_id', '')}\n"
            f"Координаты: {get('coordinate', '')}\n"
            f"Описание: {get('description', '')}"
        )

    df["__blob__"] = df.apply(row_text, axis=1).astype(str).str.slice(0, TRIM_CHARS)
    logging.info(f"✓ Excel загружен: {len(df)} объектов")
    return df

DB = load_db()

# -------- Эмбеддинги --------
def tfidf_embed(text: str) -> np.ndarray:
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
            logging.warning(f"Ошибка кэша: {e}")
            try:
                os.remove(EMB_CACHE)
            except:
                pass

    texts = df["__blob__"].tolist()
    titles = df["title"].tolist()
    ids = df["id"].astype(int).to_numpy()
    
    logging.info("↳ Создаю эмбеддинги...")
    vecs = np.array([tfidf_embed(t) for t in texts], dtype=np.float32)
    blobs = np.array(texts, dtype=object)
    titles_arr = np.array(titles, dtype=object)
    
    np.savez_compressed(EMB_CACHE, ids=ids, vecs=vecs, blobs=blobs, titles=titles_arr)
    logging.info("✓ Эмбеддинги сохранены")
    return ids, vecs, blobs, titles_arr

IDS, VECS, BLOBS, TITLES = build_or_load_embeddings(DB)

# -------- Поиск мест --------
def search_top_k(q: str, top_k=TOP_K, min_sim=MIN_SIM) -> List[Tuple]:
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
    
    sorted_results = sorted(results_dict.items(), key=lambda x: (-x[1][2] < 0, x[1][2], -x[1][0]))
    
    context, total = [], 0
    for i, (sim, _, _) in sorted_results[:top_k * 3]:
        blob = BLOBS[i]
        if total + len(blob) > CTX_CHAR_BUDGET:
            break
        context.append((float(sim), int(IDS[i]), blob, TITLES[i]))
        total += len(blob)
    
    return context[:top_k]

def make_context_block(hits: List[Tuple]) -> str:
    blocks = []
    for i, (sim, _id, blob, title) in enumerate(hits, 1):
        blocks.append(f"{i}. {title}\n{blob}\n")
    return "\n".join(blocks)

# -------- API запрос --------
async def ask_gpt_async(query: str, context: str = "", system_prompt: str = None, current_time: str = None) -> str:
    if context:
        user_msg = f"На основе информации, ответь на вопрос. ОБЯЗАТЕЛЬНО укажи точный адрес.\n\n{context}\n\nВопрос: {query}"
    else:
        user_msg = query
    
    if not system_prompt:
        system_prompt = "Ты помощник туриста по Нижнему Новгороду. ВСЕГДА указывай точный адрес. БЕЗ ТАБЛИЦ, без маркеров, без форматирования."
    
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_CHAT,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.5,
        "top_p": 0.9,
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OPENROUTER_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            answer = result["choices"][0]["message"]["content"].strip()
                            answer = re.sub(r'\|[^\n]*\|', '', answer)
                            answer = re.sub(r'[-]{3,}', '', answer)
                            answer = re.sub(r'#+\s+', '', answer)
                            answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
                            answer = re.sub(r'^[\•\-\*]\s+', '— ', answer, flags=re.MULTILINE)
                            answer = re.sub(r'\n\n\n+', '\n\n', answer)
                            answer = re.sub(r' +', ' ', answer)
                            return answer.strip()
                    else:
                        logging.warning(f"HTTP {response.status}")
                        if response.status == 429 and attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                            continue
                        elif response.status in [500, 502, 503, 504] and attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY)
                            continue
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logging.warning(f"Ошибка: {e}")
                await asyncio.sleep(RETRY_DELAY)
                continue
            raise
    raise Exception("Все попытки исчерпаны")

# -------- Telegram --------
@bot.message_handler(commands=['start'])
async def start(message):
    user_id = message.chat.id
    
    try:
        await bot.delete_message(user_id, message.message_id)
    except:
        pass
    
    bot_msg = await bot.send_message(user_id,
        "🏛️ Привет! Я AI-помощник туриста по Нижнему Новгороду.\n\n"
        "Я помогу составить персональный маршрут прогулки!\n\n"
        "📝 Ответь на 3 вопроса:\n\n"
        "1️⃣ Что тебя интересует?\n"
        "(например: история, архитектура, музеи, кофейни, стрит-арт)"
    )
    
    user_states[user_id] = {
        'step': 1,
        'bot_msg_id': bot_msg.message_id
    }
    
    logging.info(f"User {user_id}: /start")

@bot.message_handler(commands=['help'])
async def help_cmd(message):
    try:
        await bot.delete_message(message.chat.id, message.message_id)
    except:
        pass
    
    await bot.send_message(message.chat.id,
        "ℹ️ Я умею:\n"
        "✅ Составлять персональный маршрут\n"
        "✅ Подбирать места по интересам\n"
        "✅ Учитывать время и местоположение\n\n"
        "Напиши /start чтобы начать! 🚀"
    )

@bot.message_handler(content_types=['text'])
async def handle(message):
    try:
        user_id = message.chat.id
        user_msg_id = message.message_id
        q = message.text.strip()
        
        if not q or len(q) < 1:
            return
        
        logging.info(f"User {user_id} | Step {user_states.get(user_id, {}).get('step', 0)} | Q: {q}")
        
        if user_id not in user_states:
            await bot.send_message(user_id, "Напиши /start чтобы начать!")
            return
        
        state = user_states[user_id]
        step = state.get('step', 0)
        bot_msg_id = state.get('bot_msg_id')
        
        try:
            await bot.delete_message(user_id, user_msg_id)
        except:
            pass
        
        # ШАГ 1: Интересы
        if step == 1:
            state['interests'] = q
            state['step'] = 2
            logging.info(f"User {user_id}: Интересы - {q}")
            
            # Редактируем сообщение бота
            try:
                await bot.edit_message_text(
                    f"✅ Интересы: {q}\n\n"
                    "2️⃣ Сколько времени у тебя есть?\n"
                    "(укажи в часах, например: 2 или 1.5)",
                    user_id,
                    bot_msg_id
                )
            except:
                new_msg = await bot.send_message(user_id,
                    f"✅ Интересы: {q}\n\n"
                    "2️⃣ Сколько времени у тебя есть?\n"
                    "(укажи в часах, например: 2 или 1.5)"
                )
                state['bot_msg_id'] = new_msg.message_id
            return
        
        # ШАГ 2: Время
        elif step == 2:
            try:
                time_str = q.lower()
                time_str = time_str.replace('часа', '').replace('часов', '').replace('час', '')
                time_str = time_str.replace('ч', '').replace(',', '.').strip()
                time_hours = float(time_str.split()[0] if time_str.split() else time_str)
                
                if time_hours <= 0 or time_hours > 12:
                    raise ValueError("Время от 0.5 до 12 часов")
                
                state['time'] = time_hours
                state['step'] = 3
                logging.info(f"User {user_id}: Время - {time_hours}ч")
                
                # Редактируем сообщение бота
                try:
                    await bot.edit_message_text(
                        f"✅ Интересы: {state['interests']}\n"
                        f"✅ Время: {time_hours} ч\n\n"
                        "3️⃣ Где ты сейчас?\n"
                        "(адрес или ориентир, например: 'Кремль', 'ТЦ Небо')",
                        user_id,
                        bot_msg_id
                    )
                except:
                    new_msg = await bot.send_message(user_id,
                        f"✅ Интересы: {state['interests']}\n"
                        f"✅ Время: {time_hours} ч\n\n"
                        "3️⃣ Где ты сейчас?\n"
                        "(адрес или ориентир, например: 'Кремль', 'ТЦ Небо')"
                    )
                    state['bot_msg_id'] = new_msg.message_id
                return
            except Exception as e:
                logging.warning(f"User {user_id}: Ошибка парсинга времени '{q}' - {e}")
                try:
                    await bot.edit_message_text(
                        f"✅ Интересы: {state['interests']}\n\n"
                        "2️⃣ ⚠️ Укажи время числом (например: 2 или 1.5)",
                        user_id,
                        bot_msg_id
                    )
                except:
                    pass
                return
        
        # ШАГ 3: Локация и маршрут
        elif step == 3:
            state['location'] = q
            logging.info(f"User {user_id}: Локация - {q}")
            
            interests = state['interests']
            time_h = state['time']
            location = state['location']
            
            # Получаем текущее время и дату
            now = datetime.now()
            current_time_str = now.strftime("%d.%m.%Y %H:%M")
            current_hour = now.hour
            
            logging.info(f"User {user_id}: Текущее время - {current_time_str}")
            
            # Редактируем сообщение на заглушку
            try:
                await bot.edit_message_text(
                    f"✅ Интересы: {interests}\n"
                    f"✅ Время: {time_h} ч\n"
                    f"✅ Локация: {location}\n\n"
                    "🔄 Генерирую персональный маршрут...\n"
                    "Подбираю места, анализирую маршрут...\n\n"
                    "⏳ Это займёт несколько секунд",
                    user_id,
                    bot_msg_id
                )
            except:
                pass
            
            if time_h <= 2:
                places_count = 3
            elif time_h <= 4:
                places_count = 4
            else:
                places_count = 5
            
            logging.info(f"User {user_id}: Поиск {places_count} мест")
            hits = search_top_k(interests, top_k=places_count + 5)
            
            if not hits:
                await bot.edit_message_text(
                    "❌ Не нашёл подходящих мест. Попробуй изменить интересы.\n\n"
                    "Напиши /start для нового маршрута.",
                    user_id,
                    bot_msg_id
                )
                state['step'] = 0
                return
            
            logging.info(f"User {user_id}: Найдено {len(hits)} мест")
            ctx = make_context_block(hits)
            
            route_prompt = (
                f"Составь маршрут прогулки по Нижнему Новгороду:\n\n"
                f"Интересы: {interests}\n"
                f"Время прогулки: {time_h} ч\n"
                f"Начальная точка: {location}\n"
                f"Текущее время и дата: {current_time_str}\n"
                f"Текущий час: {current_hour}:00\n\n"
                f"Задача:\n"
                f"1. Выбери {places_count} места\n"
                f"2. Для каждого объясни ПОЧЕМУ оно подходит под интересы '{interests}'\n"
                f"3. Укажи ТОЧНЫЙ АДРЕС\n"
                f"4. Предложи порядок посещения\n"
                f"5. Укажи РАЗНОЕ время на каждое место (от 30 до 90 мин)\n"
                f"6. БЕЗ ТАБЛИЦ! Только простой текст\n\n"
                f"Формат:\n"
                f"Место 1: Название\n"
                f"Адрес: ...\n"
                f"Почему подходит: ...\n"
                f"Время: ~XX мин\n\n"
                f"В конце — краткий таймлайн ТЕКСТОМ (не таблицей).\n"
                f"Таймлайн должен быть привязан к текущему времени {current_time_str}.\n"
                f"Например если текущее время 14:30, то:\n"
                f"14:30-15:10 — Место 1 (40 мин)\n"
                f"15:10-15:25 — Переход и отдых (15 мин)\n"
                f"И так далее."
            )
            
            system_prompt = (
                "Ты гид по Нижнему Новгороду. Составляешь ПЕРСОНАЛЬНЫЕ маршруты. "
                "ОБЯЗАТЕЛЬНО объясняй выбор каждого места. ВСЕГДА указывай адреса. "
                "НЕ ИСПОЛЬЗУЙ ТАБЛИЦЫ! Только простой текст. НЕ ИСПОЛЬЗУЙ СИМВОЛЫ | и ---. "
                "Таймлайн должен быть с конкретным времени (ЧЧ:ММ - ЧЧ:ММ)."
            )
            
            try:
                answer = await ask_gpt_async(route_prompt, ctx, system_prompt, current_time_str)
                
                # Заменяем на готовый маршрут
                await bot.edit_message_text(
                    f"🗺️ Твой персональный маршрут готов!\n\n{answer}\n\n"
                    f"Хорошей прогулки! 🚶\n\nНапиши /start для нового маршрута.",
                    user_id,
                    bot_msg_id
                )
                
                logging.info(f"✓ User {user_id}: Маршрут отправлен ({len(answer)} симв)")
            except Exception as e:
                logging.error(f"User {user_id}: Ошибка GPT - {e}")
                await bot.edit_message_text(
                    "⚠️ Сервис перегружен, попробуй позже.\n\n"
                    "Напиши /start для нового маршрута.",
                    user_id,
                    bot_msg_id
                )
            
            state['step'] = 0
    
    except Exception as e:
        logging.error(f"User {message.chat.id}: Критическая ошибка - {e}")
        await bot.send_message(message.chat.id, "❌ Ошибка. Напиши /start")

# -------- Main --------
if __name__ == "__main__":
    logging.info("🤖 AI-помощник туриста запущен!")
    try:
        asyncio.run(bot.polling(non_stop=True, interval=0))
    except KeyboardInterrupt:
        logging.info("Бот остановлен")
