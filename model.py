from transformers import BertForSequenceClassification, TrainingArguments, Trainer
from transformers import BertTokenizer, DataCollatorWithPadding
import torch

class Model:
    def __init__(self, data_collator):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.data_collator = data_collator

    def train(self, train_dataset):
        training_args = TrainingArguments(
            per_device_train_batch_size=8,
            output_dir='./results',
            num_train_epochs=3,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()


