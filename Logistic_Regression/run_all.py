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
    # scripts = ["Logistic_Regression/Genre/Logistic_Regression_Genre.py", "Logistic_Regression/predict_genre.py"]

    # Uncomment for Emotion Classification
    # WARNING: Program takes at least 10 minutes to finish running
    print("Running model for emotion")
    scripts = ["Logistic_Regression/Emotion/Logistic_Regression_Emotion.py", "Logistic_Regression/predict_emotion.py"]
    
    for script in scripts:
        run_script(script)
