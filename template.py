import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

project_name = 'Wine Quality Prediction'

list_of_files = [
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/utils/common.py',
    f'src/{project_name}/config/__init__.py',
    f'src/{project_name}/config/config.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/entities/__init__.py',
    f'src/{project_name}/constants/__init__.py',
    'config/config.yaml',
    'params.yaml',
    'schema.yaml',
    'main.py',
    'app.py',
    'requirements.txt',
    'setup.py',
    'research/research.ipynb',
    'templates/index.html',
]

for file in list_of_files:
    file_path = Path(file)

    file_dir,filename = os.path.split(file_path)

    if file_dir != '':
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f'created directory: {file_dir} for file: {filename}')

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,'w') as f:
            pass
            logging.info(f'create empty file: {file_path}')
    else:
        logging.info(f'file already exists: {file_path}')
