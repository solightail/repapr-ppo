import time
from datetime import datetime

start = datetime.now()
print(start.time())
time.sleep(3)
end = datetime.now()
print(end.time())

td = end-start

print(td)