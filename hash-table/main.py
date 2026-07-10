# В данной реализации если что то где то привысит длину массива, то всё сломается. Также в след реализации можно сделать увеличение массива

class HashMap():
    def __init__(self):
        self.table: list[None] = [None] * 16
        
    def put(self, key, value):
        # переводим в хэш и делим на остаток 
        index: int = hash(key) % 16 # Делается для того, чтобы индекс не превышал длину таблицы
        
        # Именно изменение элемента, если он там уже есть
        if self.table[index] != None and self.table[index][0] == key:
            self.table[index] = (key, value)
            return True
        
        # В случае нового элемента добавляем его, а не изменяем
        # проверяем есть ли по индексу еще что то
        while self.table[index] != None:
            index += 1
        
        # Нашли свободное место и добавили
        self.table[index] = (key, value)
        return True
        
    def get(self, key):
        index = hash(key) % 16
        
        # Если не найден
        if self.table[index] == None or self.table[index] == "deleted":
            print("Not found")
            return True
        
        # Если мы прошлись по ряду и выскочило None
        while self.table[index][0] != key:
            if self.table[index] == None:
                print("Not found")
                return True
            index += 1
        
        # Во всех остальных случаях просто выводим значение
        print(self.table[index][1])
        return True
    
    
    def delete(self, key):
        index = hash(key) % 16
        while self.table[index][0] != key:
            if self.table[index] == None:
                print("Not found")
                return True
            index += 1
            
        self.table[index] = "delete"
        
        return True

# Тесты
map = HashMap()

map.put("banana", 1)
print(map.table)
map.delete("banana")
print(map.table)

