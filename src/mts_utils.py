import os
import time
import traceback
from pathlib import Path

def get_TimeElapsed(startTime):
    endTime = time.time()
    timeElapsed = endTime - startTime
    hours, remainder = divmod(timeElapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    timeElapsedStr = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    return timeElapsedStr



def rename_folder(folder_path, new_folder_name):
    folder_path = Path(folder_path).as_posix()
    new_folder_name = Path(os.path.join(os.path.dirname(folder_path), new_folder_name)).as_posix()

    print("Renaming folder...")
    # check if folder exists
    if not os.path.exists(folder_path):
        print("Folder does not exist")
        return
    
    # check if folder can be renamed
    scriptText = f"""
import os
import time
import traceback

print("=== Renaming folder script ===")
print(f"Folder path: {folder_path}")
print(f"New folder name: {new_folder_name}")

print("\\nWaiting for 5 seconds...")
time.sleep(5)

while True:
    try:
        print("Renaming folder...")
        os.rename("{folder_path}", "{new_folder_name}")
        print("Folder renamed successfully")
        print("Old folder name:  {folder_path}")
        print("New folder name:  {new_folder_name}")
        break
    except FileExistsError:
        print("Folder already exists")
        new_folder_name = input("Enter a new folder name: ")
        continue
    except WindowsError:
        print("Folder cannot be renamed")
        # wait for user input, try again
        input("Close programs that may be using the folder and press Enter to try again.")
        continue
    except Exception as e:
        print("An error occurred: ", e)
        traceback.print_exc()
        break
    """

    scriptPath = os.path.join(os.getcwd(), 'rename_folder_script.py')
    with open(scriptPath, 'w') as f:
        f.write(scriptText)
    
    # execute script
    print(f"Executing script: {scriptPath}")
    os.system(f"start cmd /k python {scriptPath}")

    # remove temp file
    time.sleep(2)
    os.remove(scriptPath)
