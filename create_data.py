import requests
import os
from os.path import basename
import urllib.request as urllib2
import threading
import time


class download(threading.Thread):
    def __init__(self,directory,file,ThreadId,counter):
        threading.Thread.__init__(self)
        self.directory = directory
        self.file = file
        self.ThreadId = ThreadId
        self.counter = counter
        
    def run(self):
        name = (file.split('.')[0])
        filename = os.path.join(directory,file)
        if filename.endswith(".txt"): 
            print (name)
            with open(filename, encoding="utf-8") as file:
                count = 0
                dir = 'training_data/'+name+'/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                for l in file.readlines():
                    url = l
                    try:
                        filedata = urllib2.urlopen(url)
                        datatowrite = filedata.read()
                        with open(os.path.join(dir,name+str(count))+'.jpg', 'wb') as f:  
                            f.write(datatowrite)                
                        count +=1
                    except: 
                        pass
            continue
        else:
            continue

def download(directory,file):
    name = (file.split('.')[0])
    filename = os.path.join(directory,file)
    if filename.endswith(".txt"): 
        print (name)
        with open(filename, encoding="utf-8") as file:
            count = 0
            dir = 'training_data/'+name+'/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            for l in file.readlines():
                url = l
                try:
                    filedata = urllib2.urlopen(url)
                    datatowrite = filedata.read()
                    with open(os.path.join(dir,name+str(count))+'.jpg', 'wb') as f:  
                        f.write(datatowrite)                
                    count +=1
                except: 
                    pass
        continue
    else:
        continue
# Create two threads as follows
try:    
    directory = '.\Imagesets'
    for file in os.listdir(directory):
        thread.start_new_thread( download, (directory, file))
except:
   print ("Error: unable to start thread")
while 1:
   pass