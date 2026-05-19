from typing import Callable
        
def query_limiter(function: Callable):
    def wrapper(*args, **kwargs):
        if args[0].counter >= args[0].limit:
            print("Количество запросов достигло своего пика")
            return False
        args[0].counter += 1
        result = function(*args, **kwargs)
        print(f"Запрос номер {args[0].counter}")
        return result
    return wrapper
 
class ApiService:

    def __init__(self, limit: int):
        self.limit = limit
        self.counter = 0
    
    @query_limiter
    def get_data(self, query):
        print(f"Обработка запроса {query}")
   
service_one = ApiService(5)
service_second = ApiService(10)

while (result := service_one.get_data("test_query")) != False:
    print(result)
