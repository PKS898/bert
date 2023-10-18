import torch
from transformers import BertTokenizer, DataCollatorWithPadding

def tokenize_and_prepare_data(X_train, y_train):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    encoded_data = tokenizer.batch_encode_plus(
        X_train, 
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    
    y_train = torch.tensor(y_train)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = [{'input_ids': input_ids[i], 'attention_mask': attention_masks[i], 'labels': y_train[i]} for i in range(len(X_train))]
    return train_dataset, data_collator
