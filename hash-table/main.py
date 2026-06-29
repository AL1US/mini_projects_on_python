class HashMap():
    def __init__(self):
        self.table = [None] * 16
        

    def put(self, key, value):
        # переводим в хэш и делим на остаток 
        index = hash(key) % 16
        
        if self.table[index] != None and self.table[index][0] == key:
            self.table[index] = (key, value)
            return True
        
        # проверяем есть ли по индексу еще что то
        while self.table[index] != None:
            index += 1
        
        self.table[index] = (key, value)
        return True
        
    def get(self, key):
        index = hash(key) % 16
        
        while self.table[index][0] != key:
            if self.table[index][0] == None:
                print("Not found")
                return True
            index += 1
            
        print(self.table[index][1])
        return True
    
    
    def delete(self, key):
        pass

map = HashMap()

map.put("banana", 1)
map.put("apple", 2)

map.get("banana")
map.get("apple")
