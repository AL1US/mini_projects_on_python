array = []
for i in range(100):
    array.append(i)
    
def generator(data, page_len: int):
    start = 0
    
    while start < len(data):
        end = start + page_len
        yield data[start:end]
        
        start = end

gen = generator(array, 10)
print(next(gen))
print(next(gen))
print(next(gen))


        