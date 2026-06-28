import time
user = {}

def reg():
    name = input("Input your name: ")
    balance = int(input("Input your balance: "))
    
    user[name] = balance
    
    print(f"\nuser {name} added in dict and her balance = {balance}")
    time.sleep(2.5)
    
def get_balance():
    key = str(input("\nInput name user: "))
    
    if key not in user:
        print("Not found this user")
    else:
        print(f"\nbalance {key} = ${user[key]}")
    time.sleep(2.5)
        
while True:
    try:
        print("\n0 -> exit")
        print("1 -> create user")
        print("2 -> get user balance")
        
        choise = int(input("\nInput your choise: "))
        
        if choise == 0:
            break
        elif choise == 1:
            reg()
        elif choise == 2:
            get_balance()
    except ValueError:
        print("\nYOUR INPUT INCORECT!!!!!!!")
    except Exception as e:
        print(f"Error -> {e}")