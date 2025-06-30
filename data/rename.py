import os

path = '/mnt/localstorage/mspremulli/Model_Checkpoints/rt-1_new_dataset_continue/'

files = os.listdir(path)

       
    
for file in files: 
        
    step = file.split("-")[1].split(".")[0]
    file_name = 'model_save-{}.pt'.format(step)
    os.rename(path + file, path + file_name)