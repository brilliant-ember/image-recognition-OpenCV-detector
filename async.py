import threading
import time

def print_time( threadName, delay):
   time.sleep(delay)
   print(threadName)


t1 = threading.Thread(target=print_time, args=("Thread 1", 5))
t1.daemon = True
t2 = threading.Thread(target=print_time, args=("Thread 2", 1))
t2.daemon = True
print("ONE")
t1.start()
t2.start()
t1.join()
t2.join()
print("TWp")