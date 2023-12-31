import torch
import clip
import pandas as pd
import os.path as osp
import pickle
from operator import itemgetter
import time
import openai
import os
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY


def read_pickle(dir):
    with open(dir, 'rb') as handle:
        b = pickle.load(handle)
    return b


def write_pickle(dir, data):
    with open(dir, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Timer:
    def __init__(self):
        self.t1 = None

    @staticmethod
    def delta_to_string(td):
        res_list = []

        def format():
            return ", ".join(reversed(res_list)) + " elapsed."

        seconds = td % 60
        td //= 60
        res_list.append(f"{round(seconds, 3)} seconds")

        if td <= 0:
            return format()

        minutes = td % 60
        td //= 60
        res_list.append(f"{minutes} minutes")

        if td <= 0:
            return format()

        hours = td % 24
        td //= 24
        res_list.append(f"{hours} hours")

        if td <= 0:
            return format()

        res_list.append(f"{td} days")

        return format()

    def __enter__(self):
        self.t1 = time.time()

    def __exit__(self, *args, **kwargs):
        t2 = time.time()
        td = t2 - self.t1
        print(self.delta_to_string(td))


def top_n(input_dict, n):
    return dict(sorted(input_dict.items(), key=itemgetter(1), reverse=True)[:n])


def find_products(text_input, category_df, image_pickle_path):
    text_input = [text_input]

    # stage one, compare categories
    category_df = category_df[~category_df["encoded_category"].isna()]
    categories = list(category_df["category"].values)

    categories_features = torch.stack(list(category_df["encoded_category"].values))
    encoded_texts = clip.tokenize(text_input, truncate=True).to(device)

    with torch.no_grad():
        text_features = model.encode_text(encoded_texts)
        categories_features /= categories_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = 100 * categories_features @ text_features.T

    res = dict(zip(categories, similarity.reshape(-1).tolist()))

    res = sorted(res.items(), key=itemgetter(1), reverse=True)

    n = 10
    res = res[:n]
    res_set = set([r[0] for r in res])

    # do image matching
    res = []
    for cat in res_set:
        store_path = osp.join(image_pickle_path, f"{cat}.pkl")
        cat_res = read_pickle(store_path)
        res.append(cat_res)
    res = pd.concat(res, axis=0)

    uniq_ids = list(res["uid"].values)
    image_features = torch.stack(list(res["encoded_image"].values))
    similarity = 100 * image_features @ text_features.T
    res = dict(zip(uniq_ids, similarity.reshape(-1).tolist()))
    res = sorted(res.items(), key=itemgetter(1), reverse=True)

    n = 5
    res = res[:n]
    res_set = set([r[0] for r in res])

    return res_set


def load_data(pickle_path):
    category_df = read_pickle(osp.join(pickle_path, "categories.pkl"))
    meta_df = read_pickle(osp.join(pickle_path, "meta_data.pkl"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    return device, model, preprocess, category_df, meta_df


pickle_path = "data/pickle"
image_pickle_path = "data/image_pickle"

with Timer():
    (
        device,
        model,
        preprocess,
        category_df,
        meta_df
    ) = load_data(pickle_path)

messages = []

res_list = []

prefix = (
    "You are a chatbot that helps user find furniture they are looking for."
    "Our product database contains the following information about furniture: "
    "1. Color"
    "2. Price range"
    "3. Material"
    "4. Room where the furniture will be placed."

    "Based on the the information the user has provided thus far as well as the user's message below, do you have enough information to find an appropriate product from the dataset?"

    "If not, ask the user questions that will help you find an appropriate product from the dataset. Otherwise, summarize exactly what the user is looking for."

    "Here is the user's message: "
)

def get_response(message):
    if message:
        print(f"User entered: {message}")
        messages.append(
            {"role": "user", "content": f"{prefix} {message}"},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )

        bot_reply = chat.choices[0].message.content

        def is_question(sentence):
            return "?" in sentence

        print(f"Was message a question: {str(is_question(bot_reply))}")

        print(f"ChatGPT: {bot_reply}")
        messages.append({"role": "assistant", "content": bot_reply})

        if is_question(bot_reply):
            return bot_reply, None
        else:
            print("looking for products...")
            result = find_products(bot_reply, category_df, image_pickle_path)
            print("found products")
            return bot_reply, result
