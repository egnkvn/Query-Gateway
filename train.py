import joblib
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def main():
    print("Loading dataset...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    filtered_ds = ds.filter(lambda x: x["category"] in ["classification", "summarization", "creative_writing", "general_qa"])
    split_ds = filtered_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]
    
    def prepare_data(dataset):
        texts = []
        labels = []
        fast_categories = ["classification", "summarization"]
        slow_categories = ["creative_writing", "general_qa"]
        for item in dataset:
            texts.append(item["instruction"])
            if item["category"] in fast_categories:
                labels.append(0)
            elif item["category"] in slow_categories:
                labels.append(1)
        return texts, labels

    train_texts, train_labels = prepare_data(train_ds)
    test_texts, test_labels = prepare_data(test_ds)
    print(f"Training data: {len(train_texts)}, Test data: {len(test_texts)}")

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    train_embeddings = encoder.encode(train_texts, show_progress_bar=True)
    test_embeddings = encoder.encode(test_texts, show_progress_bar=True)

    # Train
    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(train_embeddings, train_labels)

    # Save
    joblib.dump(classifier, 'classifier.joblib')

    # Evaluate
    loaded_classifier = joblib.load('classifier.joblib')
    predictions = loaded_classifier.predict(test_embeddings)
    print(classification_report(test_labels, predictions, target_names=["Fast Path (0)", "Slow Path (1)"]))

    

if __name__ == "__main__":
    main()