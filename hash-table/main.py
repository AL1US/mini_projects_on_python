# В данной реализации если что то где то привысит длину массива, то всё сломается. Также в след реализации можно сделать увеличение массива
class HashMap():
    def __init__(self):
        self.table: list[None] = [None] * 16
        
    def put(self, key, value):
        index: int = hash(key) % 16

        while self.table[index] != None:
            if self.table[index] != "deleted" and self.table[index][0] == key:
                self.table[index] = (key, value)
                return True

            index += 1

        self.table[index] = (key, value)
        return True
        
    def get(self, key):
        index = hash(key) % 16

        while self.table[index] != None:
            if self.table[index] != "deleted" and self.table[index][0] == key:
                print(self.table[index][1])
                return True

            index += 1

        print("Not found")
        return True
    
    def delete(self, key):
        index = hash(key) % 16

        while self.table[index] != None:
            if self.table[index] != "deleted" and self.table[index][0] == key:
                self.table[index] = "deleted"
                return True

            index += 1

        print("Not found")
        return True
        
# Тесты
map = HashMap()

map.put("banana", 1)
print(map.table)
map.delete("banana")
print(map.table)

