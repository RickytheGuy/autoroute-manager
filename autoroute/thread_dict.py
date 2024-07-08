import threading
import os
import json

class ThreadSafeDict:
    def __init__(self, file: str):
        self.lock = threading.Lock()
        self.data = {}
        self.file = file
        if os.path.exists(file):
            with open(file, 'r') as f:
                self.data = json.load(f)

    def set(self, key, value):
        with self.lock:
            self.data[key] = value

    def get(self, key, default=None):
        with self.lock:
            return self.data.get(key, default)

    def remove(self, key):
        with self.lock:
            if key in self.data:
                del self.data[key]   
                
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __repr__(self):
        return repr(self.data)
    
    def save(self):
        with self.lock:
            with open(self.file, 'w') as f:
                json.dump(self.data, f, indent=4)