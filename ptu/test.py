'''
from tkinter.ttk import LabeledScale
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

ARTICLE_TO_SUMMARIZE = "My friends carbs."
ARTICLE_TO_SUMMARIZE_1 = "My friends are cool but they eat too many carbs."
LABLE = "My friends carbs. I love them."
LABLE_1 = "My friends are cool but they eat too many carbs. I love them."

print([ARTICLE_TO_SUMMARIZE,ARTICLE_TO_SUMMARIZE_1])
print([LABLE,LABLE_1])
inputs = tokenizer([ARTICLE_TO_SUMMARIZE,ARTICLE_TO_SUMMARIZE_1], truncation=True, padding=True, return_tensors='pt')
labels = tokenizer([LABLE,LABLE_1], truncation=True, padding=True, return_tensors='pt')

Seq2seq_output=model(input_ids = inputs['input_ids'],
                      attention_mask = inputs['attention_mask'],
                      labels = labels['input_ids'])

print(Seq2seq_output[0])
'''
# Generate Summary
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

ARTICLE_TO_SUMMARIZE_1 = "My friends are cool but they eat too many carbs."
ARTICLE_TO_SUMMARIZE_2 = "My friends are cool."
ARTICLE_TO_SUMMARIZE_3 = "They eat too many carbs."
inputs = tokenizer([ARTICLE_TO_SUMMARIZE_1,ARTICLE_TO_SUMMARIZE_2,ARTICLE_TO_SUMMARIZE_3],truncation=True, padding=True, return_tensors='pt')

# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=30, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
