#download and unzip data from gdrive.
#download and unzip data from gdrive.
import subprocess


def get_data_dvc():
    dvc_command = "dvc pull <data_folder>.dvc"

    result = subprocess.run(dvc_command, shell=True)
    if result.returncode != 0:
        print("Error occurred while pulling data with DVC")
    else:
        print("Data pulled successfully with DVC")


if __name__ == "__main__":
    get_data_dvc()
