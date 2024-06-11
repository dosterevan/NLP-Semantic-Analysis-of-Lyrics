import subprocess

def run_script(script_name):
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
    else:
        print(f"Successfully ran {script_name}:\n{result.stdout}")

if __name__ == "__main__":
    # Uncomment for Genre Classification
    # print("Running model for genre")
    # scripts = ["SVM/Preprocessing/pre_processing.py", "SVM/feature_extraction/feature_extraction.py", "SVM/train/train_model_genre.py", "SVM/validation/evaluate_model_genre.py", "SVM/predict_genre.py"]
    
    # Uncomment for Emotion Classification
    print("Running model for emotion")
    scripts = ["SVM/Preprocessing/pre_processing.py", "SVM/feature_extraction/feature_extraction.py", "SVM/train/train_model_seed.py", "SVM/validation/evaluate_model_seed.py", "SVM/predict_emotion.py"]
    for script in scripts:
        run_script(script)
