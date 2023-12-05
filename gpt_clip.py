import torch
import clip
import pandas as pd
import os.path as osp
import pickle
from operator import itemgetter
import time
import openai
import os
from dotenv import load_dotenv, find_dotenv

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

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

#ADDED-12/4
loadfile = CSVLoader(file_path='/data/DescriptionExample.csv')

def retrieve_info(query,loadfile):
  data = loadfile.load()
  embeddings = OpenAIEmbeddings()
  #vetorizing and creating embedding using open source from Meta - FAISS
  db = FAISS.from_documents(data, embeddings)
  #getting 3 top results that are similar
  similar_response = db.similarity_search(query, k=3)

  page_contents_array = [doc.page_content for doc in similar_response]
  # print(page_contents_array)

  return page_contents_array

def llm_initiate():
  llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
  template="""
Interpret the user needs by understanding the input along with metadata.


Below is the example of user needs:
{description}

Here is how we want to respond:
{metadata}


Provide a response by explaining in one or two sentence what the user wants :
You are looking for...

Response:
"""
  prompt = PromptTemplate(
    input_variables=["description", "metadata"],
    template=template)


  chain = LLMChain(llm=llm, prompt=prompt)
  return chain


def generate_description_for_clip(description):
    chain=llm_initiate()
   #step 1 - does similarity search
    metadata = retrieve_info(description,loadfile)

    #step 2 - puts the similar best practice in the chain model
    response=chain.run({'description': description,'metadata': metadata})
    #return response, metadata_dict
    return response


messages = []

res_list = []
prefix_default = (
    "Our product database contains the following 3 information about furniture: "
    "1. Color"
    "2. Material"
    "3. Room where the furniture will be placed."
    "You are an interpretor of user input related to furniture"
    
    "Did user provide information on 2 our of 3 information types below?"
    
    "ONLY RESPOND WITH YES OR NO"
    "RESPONSE: YES if you have more than 2 type of information to find an appropriate product from the dataset (e.g. if you have color and material info, response should be Yes)"
    "RESPONSE: NO if you don't have enough information about the furniture"
    
    "Here is the user's message: "
)

prefix_question=(   "You are a chatbot that helps user find furniture they are looking for."
    "Do not mention user, you are a end user facing chatbot. Talk like an agent"
    "Our product database contains the following information about furniture: "
    "1. Color"
    "2. Material"
    "3. Room where the furniture will be placed."

    "Respond with below:"
    "Summarize information that user has input like 'You are looking for a black chair'"
    "DO NOT MAKE UP ANY INFORMATION, THAT IS NOT EXPLICITLY PROVIDED"
    "Then, ask questions asking to provide additional information listed above, if not already provided by the user. (e.g. Can you help provide the color and material you are looking for?)"

    "Here is the user's message: "
)




def get_response(message, prefix=prefix_default):
    num_question=0
    if message:
        print(f"User entered: {message}")
        messages.append(
            {"role": "user", "content": f"{prefix} {message}"},
        )
        client = OpenAI(api_key=openai.api_key,)
        chat = client.chat.completions.create( messages=messages,model="gpt-3.5-turbo",)
        #this should be a yes or no reply, so not outputing in the UI or storing in messages.append
        bot_reply = chat.choices[0].message.content
        

        needs=message
        prefix=prefix_default
        

        def generate_result(confirm_message,reply):
          if 'yes' in str(confirm_message).lower() :
            print("looking for products...")
            result = find_products(reply, category_df, image_pickle_path)
            print("found products")
            print(result)
            return reply, result
          else:
            print("Can you help clarify and refine information on what you are looking for?")
            prefix_correction=(   "You are a chatbot that helps user find furniture they are looking for."
            
              "Do not mention user, you are a end user facing chatbot. Talk like an agent"
              "Our product database contains the following information about furniture: "
              "1. Color"
              "2. Material"
              "3. Room where the furniture will be placed."

            "The user has mentioned initially provided information is incorrect"
              "Respond with two things:"
              "First, Summarize information that user has input like 'You are looking for a black chair'"
              "Secondly, then, ask what information needs to be corrected or added and ask for correction"
              "Make sure to include words expicitly around 'to correct' or 'add' to already provided information"
              "DO NOT MAKE UP ANY INFORMATION, THAT IS NOT EXPLICITLY PROVIDED"
              "Here is the user's message: ")
            #print(f"ChatGPT: If no additional info is needed, respond with yes, indicating provided info is correct")
            #messages.append({"role": "assistant", "content": reply})
            get_response(prefix_correction, reply)
            
        
        def confirm(bot_reply, needs,num_question):
          messages.append({"role": "assistant", "content": bot_reply})
          
          if 'YES' in bot_reply or num_question>1:
            reply=generate_description_for_clip(needs)
            print(f"ChatGPT: Can you please confirm this is what you are looking for with yes or no:")
            print(f"ChatGPT: {reply}")
            messages.append({"role": "assistant", "content": reply})

            #JJ-NEED HELP TO UPDATE UI - how to take in user input info whether yes or no into confirm_message variable
            confirm_message=input(reply)
            messages.append({"role": "assistant", "content": confirm_message})
            generate_result(confirm_message,reply)
            
            
          elif num_question<=1:
            messages.append(
            {"role": "user", "content": f"{prefix_question} {needs}"},)
            chat = client.chat.completions.create( messages=messages,model="gpt-3.5-turbo",)
            bot_reply1 = chat.choices[0].message.content            
            print(f"ChatGPT: {bot_reply1}")
            messages.append({"role": "assistant", "content": bot_reply1})

            #JJ-NEED HELP TO UPDATE UI - how to take in user input info as data_asset variable
            #Even if i am appending to message, think i need to take in additioal input info?

            data_asset=input(bot_reply1)
            
            num_question+=1
            

            needs=str(needs) + ' in ' +' '+ str(data_asset)

            messages.append(
              {"role": "user", "content": f"{prefix} {needs}"},)
            chat = client.chat.completions.create( messages=messages,model="gpt-3.5-turbo",)

            #this should be a yes or no reply, so not outputing in the UI or storing in messages.append
            bot_reply2 = chat.choices[0].message.content
            confirm(bot_reply2, needs,num_question)

            #return bot_reply, None
        confirm(bot_reply, needs,num_question)


#Archive previous version
# prefix = (
#     "You are a chatbot that helps user find furniture they are looking for."
#     "Our product database contains the following information about furniture: "
#     "1. Color"
#     "2. Price range"
#     "3. Material"
#     "4. Room where the furniture will be placed."

#     "Based on the the information the user has provided thus far as well as the user's message below, do you have enough information to find an appropriate product from the dataset?"

#     "If not, ask the user questions that will help you find an appropriate product from the dataset. Otherwise, summarize exactly what the user is looking for."

#     "Here is the user's message: "
# )

# def get_response(message):
#     if message:
#         print(f"User entered: {message}")
#         messages.append(
#             {"role": "user", "content": f"{prefix} {message}"},
#         )
#         chat = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo", messages=messages
#         )

#         bot_reply = chat.choices[0].message.content

#         def is_question(sentence):
#             return "?" in sentence

#         print(f"Was message a question: {str(is_question(bot_reply))}")

#         print(f"ChatGPT: {bot_reply}")
#         messages.append({"role": "assistant", "content": bot_reply})

#         if is_question(bot_reply):
#             return bot_reply, None
#         else:
#             print("looking for products...")
#             result = find_products(bot_reply, category_df, image_pickle_path)
#             print("found products")
#             return bot_reply, result
