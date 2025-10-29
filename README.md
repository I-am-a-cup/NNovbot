# 🏛️ NNovBot - AI Tourist Assistant for Nizhny Novgorod

AI-powered Telegram bot that creates personalized walking routes around Nizhny Novgorod based on user interests, time availability, and current location.

## ✨ Features

- **Personalized Routes**: Creates custom walking routes based on your interests (history, architecture, museums, cafes, street art, etc.)
- **Smart Time Management**: Adjusts route complexity based on available time (0.5-12 hours)
- **Location-Aware**: Starts routes from your current location or preferred starting point
- **Real-Time Scheduling**: Generates timelines tied to actual current time
- **Intelligent Search**: Uses TF-IDF embeddings and hybrid search (semantic + keyword matching) for accurate place recommendations
- **Interactive Interface**: Simple 3-step conversation flow with message editing for clean UX
- **Caching System**: Pre-computed embeddings for fast response times

## 🛠️ Tech Stack

- **Python 3.10+**
- **pyTelegramBotAPI** (telebot) - Telegram Bot API wrapper
- **pandas** - Data processing
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning (TF-IDF, cosine similarity)
- **aiohttp** - Async HTTP requests
- **python-dotenv** - Environment variables management
- **OpenRouter API** - LLM integration (GPT-based models)

## 📋 Prerequisites

- Python 3.10 or higher
- Telegram account
- OpenRouter API key ([get one here](https://openrouter.ai/))
- Excel database with cultural objects (included: `cultural_objects_mnn.xlsx`)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/NNovBot.git
cd NNovBot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**How to get tokens:**

- **Telegram Bot Token**: Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, and follow instructions
- **OpenRouter API Key**: Sign up at [openrouter.ai](https://openrouter.ai/) and create an API key

### 4. Prepare Database

Ensure `cultural_objects_mnn.xlsx` is in the project root directory. The Excel file should contain columns:
- `id` - Unique identifier
- `title` - Place name
- `description` - Detailed description
- `address` - Street address
- `coordinate` - GPS coordinates
- `category_id` - Category identifier
- `url` - Website URL (optional)

### 5. Run the Bot

```bash
python chatbot.py
```

On first run, the bot will generate embeddings cache (`embeddings_cultural_sites.npz`) - this takes a few seconds.

## 📱 Usage

1. Start chat with your bot on Telegram
2. Send `/start` command
3. Answer three questions:
   - **Interests**: What interests you? (e.g., "history", "architecture", "museums")
   - **Time**: How long do you have? (e.g., "2" or "1.5" hours)
   - **Location**: Where are you now? (e.g., "Kremlin", "TC Nebo")
4. Receive personalized route with:
   - 3-5 recommended places based on your time
   - Explanation why each place matches your interests
   - Exact addresses
   - Time allocation for each location
   - Complete timeline with real clock times

## 🗂️ Project Structure

```
NNovBot/
├── chatbot.py                          # Main bot code
├── cultural_objects_mnn.xlsx           # Database with places
├── embeddings_cultural_sites.npz       # Cached embeddings (auto-generated)
├── .env                                # Environment variables (create this)
├── .env.example                        # Example env file
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## ⚙️ Configuration

### Main Settings (in `chatbot.py`)

```python
EXCEL_PATH = "cultural_objects_mnn.xlsx"  # Path to Excel database
MODEL_CHAT = "openai/gpt-oss-20b:free"   # LLM model to use
TOP_K = 15                                # Number of places to search
MIN_SIM = 0.05                            # Minimum similarity threshold
CTX_CHAR_BUDGET = 20000                   # Context character limit
MAX_RETRIES = 5                           # API retry attempts
RETRY_DELAY = 30                          # Retry delay (seconds)
```

### Customizing Models

You can change the LLM model by modifying `MODEL_CHAT`. Available free models on OpenRouter:
- `openai/gpt-oss-20b:free`
- `meta-llama/llama-3.2-3b-instruct:free`
- `microsoft/phi-3-mini-128k-instruct:free`

See [OpenRouter models](https://openrouter.ai/models) for more options.

## 🔍 How It Works

### Search Algorithm

1. **Query Processing**: User interests are converted to TF-IDF embeddings using hash-based vectorization
2. **Hybrid Search**: 
   - Semantic search using cosine similarity
   - Keyword matching on place titles
   - Combined scoring with boosting for keyword matches
3. **Ranking**: Results ranked by relevance and fit within context budget
4. **LLM Generation**: Top places sent to GPT model to create personalized narrative

### Conversation Flow

```
User: /start
Bot: Question 1 (Interests)
User: [Answer 1]
Bot: Edits message → Question 2 (Time)
User: [Answer 2]
Bot: Edits message → Question 3 (Location)
User: [Answer 3]
Bot: Edits message → "Generating route..."
Bot: Edits message → Final personalized route
```

## 🐛 Troubleshooting

### Bot doesn't respond
- Check if `TELEGRAM_BOT_TOKEN` is correct in `.env`
- Ensure bot is running (`python chatbot.py`)
- Check logs for error messages

### API errors (429, 500, 502)
- Bot automatically retries failed requests (up to 5 times)
- If persistent, check OpenRouter service status
- Consider upgrading to paid OpenRouter tier for higher rate limits

### Wrong search results
- Adjust `MIN_SIM` threshold (lower = more results, possibly less relevant)
- Increase `TOP_K` for more options
- Check Excel data quality (descriptions, addresses)

### Embeddings cache issues
- Delete `embeddings_cultural_sites.npz` and restart bot
- Cache will regenerate automatically

## 🚢 Deployment

### Option 1: Local Server

```bash
# Use screen or tmux to keep bot running
screen -S nnovbot
python chatbot.py
# Detach: Ctrl+A, D
```

### Option 2: Systemd Service (Linux)

Create `/etc/systemd/system/nnovbot.service`:

```ini
[Unit]
Description=NNovBot Telegram Bot
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/NNovBot
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 chatbot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable nnovbot
sudo systemctl start nnovbot
sudo systemctl status nnovbot
```

### Option 3: Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "chatbot.py"]
```

Build and run:

```bash
docker build -t nnovbot .
docker run -d --name nnovbot --env-file .env nnovbot
```

### Option 4: Railway / Render

1. Push code to GitHub
2. Connect repository to [Railway.app](https://railway.app) or [Render.com](https://render.com)
3. Add environment variables in dashboard
4. Deploy automatically

### Option 5: VPS (DigitalOcean, AWS, etc.)

```bash
# Connect to VPS
ssh user@your-vps-ip

# Install Python
sudo apt update
sudo apt install python3 python3-pip git

# Clone and setup
git clone https://github.com/yourusername/NNovBot.git
cd NNovBot
pip3 install -r requirements.txt

# Create .env file
nano .env
# Paste your tokens, save (Ctrl+X, Y, Enter)

# Run with nohup
nohup python3 chatbot.py > bot.log 2>&1 &
```

## 📊 Performance

- **First run**: ~5-10 seconds (embedding generation)
- **Subsequent runs**: Instant (cached embeddings)
- **Route generation**: 5-15 seconds (depends on LLM API response time)
- **Memory usage**: ~50-100 MB
- **CPU usage**: Low (mostly waiting for API responses)

## 🔒 Security

- Never commit `.env` file to git (included in `.gitignore`)
- Keep API keys private
- Rotate tokens if exposed
- Consider using environment variables in production instead of `.env` file

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is open source and available under the MIT License.


## 🙏 Acknowledgments

- Cultural objects database sourced from Nizhny Novgorod open data
- Powered by OpenRouter AI
- Built with python-telegram-bot community support

---

**⭐ Star this repository if you find it helpful!**
