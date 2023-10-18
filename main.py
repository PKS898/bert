from model import Model
from data_processing import tokenize_and_prepare_data
from custom_collator import CustomDataCollator

from transformers import logging
# Ignore transformer warning
logging.set_verbosity_error()


# Приветствие и вопрос
print("Здравствуйте! Я готов помочь с вашими вопросами о маршрутах.")


# Инициализируем модель и data_collator
X_train = ["Пешком", "Машина", "Автобус"]
y_train = [0, 1, 2]

train_dataset, data_collator = tokenize_and_prepare_data(X_train, y_train)
model = Model(data_collator)
model.train(train_dataset)

while True:
    user_question = input("Ваш вопрос (или 'выход' для завершения): ")
    
    if user_question.lower() == 'выход':
        print("До свидания!")
        break

    # Предсказываем рекомендуемый маршрут
    recommended_route = model.predict(user_question)
    print(f"Рекомендуемый маршрут: {recommended_route}")