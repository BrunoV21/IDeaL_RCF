import os
import gdown

def fetch_experiments():
    if os.path.exists('./experiments'):
        return
    else:
        gdown.download_folder('https://drive.google.com/drive/folders/1-8VnyKsC_d7VpejFSs5hIW_Sct5vr9DH?usp=sharing')