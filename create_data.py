import os
import queue
import urllib.request as urllib2
import threading
import time

exitFlag = 0


class downloadThread(threading.Thread):
    def __init__(self, ThreadId, name, directory, q):
        threading.Thread.__init__(self)
        self.name = name
        self.directory = directory
        self.ThreadId = ThreadId
        self.q = q
        
    def run(self):
        print ("Starting " + self.name)
        process_data(self.name, self.directory, self.q)
        print ("Exiting " + self.name)

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
    
def process_data(threadName, directory, q):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            data = q.get()
            queueLock.release()
            print ("%s processing %s" % (threadName, data))
            download(directory,data)
        else:
            queueLock.release()
            time.sleep(1)
                
# Create two threads as follows
threadList = ["Thread-1", "Thread-2", "Thread-3"]
fileList = []
dir_path = os.path.dirname(os.path.realpath(__file__))		  

directory = dir_path + '\Imagesets'

try:    
    for file in os.listdir(directory):
        fileList.append(file)
except:
   print ("Error: unable to start thread")

queueLock = threading.Lock()
workQueue = queue.Queue(10)
threads = []
threadID = 1

# Create new threads
print('Create threads')
for tName in threadList:
   thread = downloadThread(threadID, tName, directory, workQueue)
   thread.start()
   threads.append(thread)
   threadID += 1

# Fill the queue
queueLock.acquire()
for file in fileList:
   workQueue.put(file)
queueLock.release()

# Wait for queue to empty
while not workQueue.empty():
   pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
   t.join()
print ("Exiting Main Thread")