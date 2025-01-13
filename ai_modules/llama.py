from dotenv import load_dotenv
import transformers
import torch
from deep_translator import GoogleTranslator
from deep_translator import single_detection
import logging 

#Making logging's config
logging.basicConfig(filename = "logs.log",
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Llama():
    def __init__(self, query):
       # Loading environment and class variables
        load_dotenv()
        self.query = query
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = "meta-llama/Meta-Llama-3-8B"
        self.pipeline = transformers.pipeline(
                    "text-generation",
                    model = self.model_id,
                    model_kwargs={"torch_dtype" : torch.bfloat16},
                    device = self.device
                    )

    
    def generate_response(self):
      #Detecting language and translating into english
        detected_language = single_detection(self.query, api_key=None)  # `api_key=None` ile ücretsiz mod
        query_eng = GoogleTranslator(source = "auto", target = "en").translate(self.query)
        #System and user prompts
        messages = [{
        "role" : "system",
        "content" : "You are a model that optimizes suppliers' product descriptions and makes the product descriptions provided to you by users suitable for the e-commerce environment. "
                        },
        {
            "role" : "user",
            "content" :{query_eng} 
        }]

        prompt = self.pipeline.tokenizer.apply_chat_template(
                      messages, 
                      tokenize = False,
                      add_generation_prompt = True)

        terminators = [
                    self.pipeline.tokenizer.eos_token_id,
                    self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
                        prompt,
                        max_new_tokens = 256,
                        eos_token_id = self.terminators,
                        do_sample = True,
                        temperature = 0.6,
                        top_p = 0.9)
        
        resp = outputs[0]["generated_text"][len(prompt):]
        try:
            translated_query = GoogleTranslator(source = "en", target = detected_language).translate(resp)
            logging.info(f"Model Cevabi : {translated_query}")
        except Exception as e:
            logging.error("Model cevabi olusturulurken hata...")
        return translated_query
