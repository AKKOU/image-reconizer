import os

paths = './face1/'
start_from = 1
for item in os.listdir(path=paths):
    old_name = os.path.join(paths,item)
    new_name = os.path.join(paths,f"{start_from}" + os.path.splitext(item)[1])
    if(os.path.exists(os.path.splitext(item)[0])):
        pass
    else:
        os.rename(old_name,new_name)
        start_from += 1