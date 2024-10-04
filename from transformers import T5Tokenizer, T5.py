from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"  
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def categorize_email(email_text):
    input_text = f"summarize: {email_text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(inputs.input_ids, max_length=50, num_beams=4, early_stopping=True)

    category = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return category


email_text = """
Olá, gostaria de saber mais informações sobre as opções de pagamento disponíveis
para a compra dos seus produtos. Também quero saber se vocês aceitam parcelamento.
"""
category = categorize_email(email_text)
print(f"Categoria: {category}")
