from classes.accounts import User
from classes.characters import (
    Player,
    Warrior,
    Arrow,
    Mage,
    Tank
)
def main_menu():
    
    try:
        
        print("\n     Menu")
        print("---------------")
        print("0 -> exit")
        print("1 -> Profile")
        print("---------------")
        
        choise = int(input("Input your choise -> "))
        
        if choise == 0:
            print("Completion of the program")
            pass
        
        elif choise == 1:
            menu_profile()
    except ValueError:
        print("Your input incorect")
    except Exception as e:
        print(f"Error -> {e}")

def menu_profile():
    print("\n0 -> back")
    print("1 -> login")
    print("2 -> reg")
    print("3 -> view profile")
    obj = User()
    try:
        choice = int(input("\nInput your choice: "))
        
        if choice == 0:
            pass
        elif choice == 1:
            obj.log_in()
        elif choice == 2:
            obj.register()
        elif choice == 3:
            obj.get_profile()
        elif choice == 4:
            obj.logout()
        else:
            print("\nInput incorrect")
    except Exception as e:
        print(f"Error -> {e}")  
    main_menu()