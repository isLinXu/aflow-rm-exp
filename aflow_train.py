import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset


def load_data(file_path):
    """加载JSON数据并提取对话和标签"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts, labels = [], []
    for item in data:
        for conversation in item['conversations']:
            texts.append(conversation['value'])
        labels.append(item['chosen']['score'])

    return texts, labels


def tokenize_data(texts, labels, tokenizer):
    """将文本数据和标签数据封装成Dataset对象并进行Tokenize"""
    dataset = Dataset.from_dict({'text': texts, 'label': labels})

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return dataset


def train_model(train_dataset, val_dataset, model_name='bert-base-uncased', output_dir='./results', num_epochs=3):
    """加载预训练模型并训练"""
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     num_train_epochs=num_epochs,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     warmup_steps=500,
    #     weight_decay=0.01,
    #     logging_dir=f"{output_dir}/logs",
    #     logging_steps=10,
    #     evaluation_strategy="epoch"
    # )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    return model


def save_model(model, tokenizer, save_path='./model'):
    """保存模型和tokenizer"""
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


# 主流程
def main(data_path, model_name='bert-base-uncased', output_dir='./results', save_path='./model', num_epochs=10):
    # Step 1: 加载数据
    texts, labels = load_data(data_path)

    # Step 2: 数据集分割
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

    # Step 3: 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 4: Tokenize并创建训练和验证数据集
    train_dataset = tokenize_data(train_texts, train_labels, tokenizer)
    val_dataset = tokenize_data(val_texts, val_labels, tokenizer)

    # Step 5: 训练模型
    model = train_model(train_dataset, val_dataset, model_name=model_name, output_dir=output_dir, num_epochs=num_epochs)

    # Step 6: 保存模型
    save_model(model, tokenizer, save_path=save_path)


# 调用主函数
if __name__ == "__main__":
    # main(
    #     data_path='/Users/gatilin/Downloads/aflow/aflow_min_diff_0.05_train_6556_v2.json',
    #     model_name='bert-base-uncased',
    #     output_dir='./result_aflow_min_diff_0.05',
    #     save_path='./model_aflow_min_diff_0.05',
    #     num_epochs=10
    # )

    main(
        data_path='/Users/gatilin/PycharmProjects/aflow_eval/aflow/aflow_min_diff_0.05_train_6556_v2.json',
        model_name='bert-large-uncased',
        output_dir='./bert-large-uncased_result_aflow_min_diff_0.05',
        save_path='./bert-large-uncased_model_aflow_min_diff_0.05',
        num_epochs=10
    )


