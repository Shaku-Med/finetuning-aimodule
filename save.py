import zipfile
import os

# Create a zip file with all your project files
def create_project_zip():
    zip_filename = 'medzy-ai-project.zip'
    
    files_to_include = [
        'api.py',
        'config.py', 
        'model_setup.py',
        'data_processor.py',
        'medzy_dataset.json',
        'requirements.txt',
        'main.py',
        'trainer.py',
        'inference.py',
        'save.py'
    ]
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Add Python files
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file)
        
        # Add the trained model directory
        if os.path.exists('finetuned_model/final_model'):
            for root, dirs, files in os.walk('finetuned_model/final_model'):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path)
    
    print(f"Created {zip_filename}")
    return zip_filename

# Create and download the zip
zip_file = create_project_zip()

# Download the zip file
from google.colab import files
files.download(zip_file)