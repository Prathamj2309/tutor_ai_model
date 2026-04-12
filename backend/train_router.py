import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    print("Loading dataset...")
    # Load dataset, clean empty rows
    try:
        df = pd.read_csv("jee_data_all.csv")
    except FileNotFoundError:
        print("Error: jee_data_all.csv not found. Please provide the dataset.")
        return

    df = df.dropna(subset=['question', 'subject'])
    
    # Standardize subject labels to lowercase
    df['subject'] = df['subject'].str.lower().str.strip()
    valid_subjects = {'physics', 'chemistry', 'mathematics', 'math'}
    df = df[df['subject'].isin(valid_subjects)]
    
    # Normalize 'math' -> 'mathematics' to ensure consistency
    df['subject'] = df['subject'].replace('math', 'mathematics')

    print(f"Loaded {len(df)} valid questions.")

    # Load embedder
    print("Loading Sentence Transformer embedder (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode questions
    print("Encoding questions into embeddings (This may take a while)...")
    X = embedder.encode(df['question'].tolist(), show_progress_bar=True)
    y = df['subject'].tolist()

    # Split dataset
    print("Splitting dataset 80/20 with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train SVM Classifier
    print("Training LinearSVC Classifier...")
    classifier = LinearSVC(random_state=42, dual=False, max_iter=2000)
    classifier.fit(X_train, y_train)

    # Evaluate
    print("Evaluating Model...")
    y_pred = classifier.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # Save the SVM Brain
    model_filename = "minilm_router_brain.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(classifier, f)
        
    print(f"\nRouter Brain saved to {model_filename} successfully!")

if __name__ == "__main__":
    main()
