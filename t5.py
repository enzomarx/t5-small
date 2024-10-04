from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def train_model(email_texts, categories, epochs=3, batch_size=4):
    inputs = tokenizer(email_texts, return_tensors='pt', padding=True, truncation=True)
    labels = tokenizer(categories, return_tensors='pt', padding=True, truncation=True).input_ids

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(email_texts), batch_size):
            batch_inputs = {key: value[i:i + batch_size] for key, value in inputs.items()}
            batch_labels = labels[i:i + batch_size]

            outputs = model(**batch_inputs, labels=batch_labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch: {epoch + 1}, Batch: {i // batch_size + 1}, Loss: {loss.item()}")

email_texts = [
    "Preciso de ajuda para redefinir minha senha.",
    "Quais são as opções de pagamento disponíveis?"
]

categories = [
    "Solicitação de suporte para redefinição de senha.",
    "Pergunta sobre métodos de pagamento."
]

train_model(email_texts, categories)
