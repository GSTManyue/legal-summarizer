from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
import openai

# Load Pegasus tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

def summarize_text(text, use_gpt=False, api_key=None, max_length=300):
    if use_gpt and api_key:
        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal document summarizer."},
                    {"role": "user", "content": f"Summarize this judgment in {max_length} words:\n{text}"}
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error using GPT: {e}"

    # Pegasus summary
    inputs = tokenizer.encode(text, truncation=True, padding="longest", return_tensors="pt", max_length=1024)
    summary_ids = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

