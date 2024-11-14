import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import mean_squared_error, r2_score


# 新增pair-wise accuracy计算
def calculate_pairwise_accuracy(labels, predictions):
    """计算pair-wise accuracy"""
    correct_pairs = 0
    total_pairs = 0

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            # 比较实际标签和预测标签之间的相对大小
            if (labels[i] > labels[j] and predictions[i] > predictions[j]) or \
                    (labels[i] < labels[j] and predictions[i] < predictions[j]) or \
                    (labels[i] == labels[j] and predictions[i] == predictions[j]):
                correct_pairs += 1
            total_pairs += 1

    # 计算pair-wise accuracy
    return correct_pairs / total_pairs if total_pairs > 0 else 0


def load_model(model_path, tokenizer_path):
    """加载模型和tokenizer"""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def load_data(file_paths):
    """加载多个JSON文件的数据并提取文本和标签"""
    texts, labels = [], []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            for conversation in item['conversations']:
                texts.append(conversation['value'])
            labels.append(item['chosen']['score'])

    return texts, labels


def tokenize_data(texts, tokenizer):
    """将文本数据进行tokenize"""
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')


def evaluate_model(model, tokenizer, val_data_paths):
    """评估模型"""
    # Step 1: 加载验证数据
    val_texts, val_labels = load_data(val_data_paths)

    # Step 2: Tokenize验证数据
    val_encodings = tokenize_data(val_texts, tokenizer)

    # Step 3: 模型预测
    model.eval()
    with torch.no_grad():
        val_outputs = model(**val_encodings)
        val_predictions = val_outputs.logits.squeeze().tolist()

    # Step 4: 计算评估指标
    mse = mean_squared_error(val_labels, val_predictions)
    r2 = r2_score(val_labels, val_predictions)
    pairwise_accuracy = calculate_pairwise_accuracy(val_labels, val_predictions)

    # 输出评估结果
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(f'Pair-wise Accuracy: {pairwise_accuracy}')


def main(model_path, tokenizer_path, val_data_paths):
    """主流程"""
    # Step 1: 加载模型和tokenizer
    model, tokenizer = load_model(model_path, tokenizer_path)

    # Step 2: 评估模型
    evaluate_model(model, tokenizer, val_data_paths)


# 调用主函数
if __name__ == "__main__":
    main(
        # model_path='/svap_storage/gatilin/workspaces/working/aflow_exp/bert-base-uncased_model_aflow_min_diff_0.05',
        # model_path = '/svap_storage/gatilin/workspaces/working/aflow_exp/bert-base-uncased_10k_model_aflow_min_diff_0.05',
        # tokenizer_path='/svap_storage/gatilin/workspaces/working/aflow_exp/bert-base-uncased_10k_model_aflow_min_diff_0.05',
        model_path='/svap_storage/gatilin/workspaces/working/aflow_exp/codet5-large_10k_model_aflow_min_diff_0.05',
        tokenizer_path='/svap_storage/gatilin/workspaces/working/aflow_exp/codet5-large_10k_model_aflow_min_diff_0.05',
        val_data_paths=[
            '/svap_storage/gatilin/workspaces/working/aflow_exp/data/aflow/aflow_min_diff_0.1_test_456_v2.json'
        ]
    )