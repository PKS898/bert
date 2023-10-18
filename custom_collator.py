import torch

class CustomDataCollator:
    def __call__(self, features):
        input_ids = [feature[0] for feature in features]
        attention_masks = [feature[1] for feature in features]
        labels = [feature[2] for feature in features]

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.tensor(labels)
        }
