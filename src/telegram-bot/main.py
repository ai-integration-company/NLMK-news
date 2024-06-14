import os
import logging

import requests
from dotenv import load_dotenv
from telebot import TeleBot, custom_filters, types
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telebot.handler_backends import State, StatesGroup
from telebot.storage import StateMemoryStorage
import math

import db

load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')

bot = TeleBot(API_TOKEN)

DOWNLOAD_DIR = 'downloads'
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

state_storage = StateMemoryStorage()
bot = TeleBot(API_TOKEN, state_storage=state_storage)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:     [%(asctime)s] %(message)s')
default_tags = [
    "Technologies",
    "Innovations",
    "Innovations",
    "Trends",
    "Digitalization",
    "Automation",
    "Digitaltransformation",
    "Digitalsolutions",
    "Digitaltwins",
    "Digitaltwins",
    "Ai",
    "Iot",
    "Internetofthings",
    "Bigdata",
    "Blockchain",
    "Processmining",
    "Cloudtechnologies",
    "Quantumcomputing",
    "Smartcontracts",
    "Robotics",
    "Vr", "Ar", "Mr",
    "Virtualandaugmentedreality",
    "Generative",
    "Recognition",
    "Artificialintelligence",
    "Machinelearning",
    "Deeplearning",
    "Neuralnetworks",
    "Computervision",
    "Naturallanguageprocessing",
    "Reinforcementlearning",
    "Lowcode",
    "Nocode",
    "Metallurgical",
    "Steel",
    "Llm",
    "Ml",
    "Chatgpt",
    "It",
    "Cybersecurity",
    "Startups",
    "Startups",
    "Yandexgpt",
    "Llama",
    "Gpt",
    "Bert",
    "Openai",
    "Dalle",
    "Transformermodels",
    "Generativeadversarialnetworks",
    "Deepfake",
    "Machinevision",
    "Texttoimage",
    "Voicetotext",
    "Datavisualization",
    "Supplychainmanagement",
    "Procurement",
    "Supercomputers",
    "Devops",
    "Fintech",
    "Token",
    "Microservices",
    "Kubernetes",
    "Api",
    "Digitalfootprint",
    "Digitalidentification",
    "Intelligentdataanalysis",
    "Advancedanalytics",
    "Severstal",
    "Evraz",
    "Mmk",
    "Omk",
    "Nipponsteel"
]


class MyStates(StatesGroup):
    getting_info = State()
    answering_questions = State()
    adding_tags = State()
    deleting_tags = State()
    getting_weekly_news = State()
    getting_added_news = State()


def save_file(document, file_info):
    downloaded_file = bot.download_file(file_info.file_path)
    file_path = os.path.join(DOWNLOAD_DIR, document.file_name)
    with open(file_path, 'wb') as new_file:
        new_file.write(downloaded_file)


def gen_markup():
    markup = ReplyKeyboardMarkup(row_width=1)
    markup.add(KeyboardButton("Добавить теги"))
    markup.add(KeyboardButton("Удалить теги"))
    # markup.add(KeyboardButton("Добавить информацию"))
    markup.add(KeyboardButton("Получить недельные новости"))
    # markup.add(KeyboardButton("Дайджест по добавленной информации"))
    return markup


TAGS_PER_PAGE = 5


def gen_tag_markup(tags, page=0):
    markup = types.InlineKeyboardMarkup()
    start = page * TAGS_PER_PAGE
    end = start + TAGS_PER_PAGE
    for tag in list(tags)[start:end]:
        markup.add(types.InlineKeyboardButton(tag, callback_data=f"delete_{tag}"))

    total_pages = math.ceil(len(tags) / TAGS_PER_PAGE)
    pagination_buttons = []
    if page > 0:
        pagination_buttons.append(types.InlineKeyboardButton("⬅️", callback_data=f"page_{page-1}"))
    if page < total_pages - 1:
        pagination_buttons.append(types.InlineKeyboardButton("➡️", callback_data=f"page_{page+1}"))

    if pagination_buttons:
        markup.row(*pagination_buttons)

    return markup


@bot.callback_query_handler(func=lambda call: call.data.startswith("page_"))
def page_handler(call):
    page = int(call.data.split("_")[1])
    tags = db.get_user_tags(call.from_user.id)
    bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id,
                                  reply_markup=gen_tag_markup(tags, page))


def handle_keyboard_callbacks(message) -> bool:
    if message.text.lower() == 'добавить теги':
        bot.set_state(message.from_user.id, MyStates.adding_tags, message.chat.id)
        tags = db.get_user_tags(message.from_user.id)
        bot.send_message(message.chat.id, f"Ваши текущие теги: {', '.join(tags)}\nВведите новые теги через запятую на английском языке одним словом и только ПЕРВОЙ большой буквой:")
        return True
    if message.text.lower() == 'удалить теги':
        bot.set_state(message.from_user.id, MyStates.deleting_tags, message.chat.id)
        tags = db.get_user_tags(message.from_user.id)
        logger.info(f'{tags}')
        if tags:
            bot.send_message(message.chat.id, "Выберите теги для удаления:", reply_markup=gen_tag_markup(tags))
        else:
            bot.send_message(message.chat.id, "У вас нет тегов для удаления.")
        bot.set_state(message.from_user.id, MyStates.getting_info, message.chat.id)
        return True
    if message.text.lower() == 'добавить информацию':
        bot.set_state(message.from_user.id, MyStates.getting_info, message.chat.id)
        bot.send_message(message.chat.id, "Какую информацию вы хотите добавить?")
        return True
    if message.text.lower() == 'получить недельные новости':
        handle_getting_weekly_news(message)
        bot.set_state(message.from_user.id, MyStates.getting_info, message.chat.id)
        return True
    if message.text.lower() == 'дайджест по добавленной информации':
        handle_getting_added_news(message)
        return True
    return False


@bot.message_handler(state=MyStates.getting_info, content_types=['text'])
def handle_text_info(message):
    logger.info(f"User {message.from_user.id}. Processing a text message.")
    if handle_keyboard_callbacks(message):
        logger.info(f"User {message.from_user.id}. Process callback")
        return


@bot.message_handler(state=MyStates.adding_tags, content_types=['text'])
def handle_adding_tags(message):
    logger.info(f"User {message.from_user.id}. Adding tags.")
    
    def is_valid_tag(tag):
        return tag.isalpha() and tag.istitle() and ' ' not in tag

    new_tags = {tag.strip() for tag in message.text.split(',')}
    valid_tags = {tag for tag in new_tags if is_valid_tag(tag)}
    invalid_tags = new_tags - valid_tags
    
    if invalid_tags:
        bot.send_message(message.chat.id, f"Некорректные теги: {', '.join(invalid_tags)}. Пожалуйста, введите теги на английском языке, одним словом, с первой заглавной буквой.")
    else:
        current_tags = db.get_user_tags(message.from_user.id)
        current_tags.update(valid_tags)
        db.save_user_tags(message.from_user.id, current_tags)
        bot.send_message(message.chat.id, f"Теги добавлены: {', '.join(valid_tags)}", reply_markup=gen_markup())
        bot.set_state(message.from_user.id, MyStates.getting_info, message.chat.id)


@bot.callback_query_handler(func=lambda call: call.data.startswith('delete_'))
def callback_delete_tag(call):
    tag_to_delete = call.data.split('_')[1]
    user_tags = db.get_user_tags(call.from_user.id)
    if tag_to_delete in user_tags:
        user_tags.remove(tag_to_delete)
        db.save_user_tags(call.from_user.id, user_tags)
        bot.answer_callback_query(call.id, f"Тег '{tag_to_delete}' удален.")
        bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id,
                                      reply_markup=gen_tag_markup(user_tags))
    else:
        bot.answer_callback_query(call.id, f"Тег '{tag_to_delete}' не найден.")


def handle_getting_weekly_news(message):
    logger.info(f"User {message.from_user.id}. Getting weekly news.")
    tags = db.get_user_tags(message.from_user.id)
    response = requests.post("http://ml:3000/weekly_news", json={"tags": list(tags)}, timeout=600)
    if response.status_code == 200:
        news = response.json()['answer']
        bot.send_message(message.chat.id,"Новости за последнюю неделю:\n")
        for new in news:
            bot.send_message(message.chat.id,
                            new)
    else:
        bot.send_message(message.chat.id,
                         "Произошла ошибка при получении новостей.",
                         reply_markup=gen_markup())
    bot.set_state(message.from_user.id, MyStates.getting_info, message.chat.id)


def handle_getting_added_news(message):
    logger.info(f"User {message.from_user.id}. Getting added news digest.")
    tags = db.get_user_tags(message.from_user.id)
    response = requests.post("http://ml:3000/added_news", json={"tags": list(tags)})
    if response.status_code == 200:
        bot.send_message(message.chat.id,
                         "Вот ваш дайджест по добавленной информации:",
                         reply_markup=gen_markup())
    else:
        bot.send_message(message.chat.id,
                         "Произошла ошибка при получении дайджеста.",
                         reply_markup=gen_markup())


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    logger.info(f"User {message.from_user.id}. Starting the bot.")
    tags = db.get_user_tags(message.from_user.id)
    if not tags:
        db.save_user_tags(message.from_user.id, set(default_tags))
    bot.set_state(message.from_user.id, MyStates.getting_info, message.chat.id)
    bot.send_message(message.chat.id,
                     "Здравствуйте! Выберите режим, в котором вы хотите работать с помощью кнопки.",
                     reply_markup=gen_markup())


if __name__ == '__main__':
    db.init()
    bot.add_custom_filter(custom_filters.StateFilter(bot))
    bot.infinity_polling(skip_pending=True)
