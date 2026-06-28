class User:
    def __init__(self):
        self.accounts = {} # name => password, all_characters
        
        self.current_user = None
        self.current_player = None

    # log in to your account        
    def log_in(self):
        try:
            
            name = input("Input your name: ")
            if name not in self.accounts:
                print("\nNot found this user")
                return False
            
            password = input("Input your password: ")
            if  self.accounts[name]["password"] != password:
                print("\nIncorect password")
                return False
            
            self.current_user = name
            print("\nYou have successfully logged in to your account!")
            return True

        except ValueError:
            print("\nYour input is incorrect")
            return False
        except Exception as e:
            print(f"Error -> {e}")        
            
    # register your account
    def register(self):
        name = input("\nInput your name: ")
        if name in self.accounts:
            print("This account has already been created")
            return False
        password = input("Input password: ")
        
        self.current_user = name
        self.accounts[name] = {
            "password": password,
            "characters": []
        }
        self.current_user = name
        print(f"\nAccount {name} has been created successfully")
        return True
    
    def logout(self):
        self.current_user = None
        self.current_player = None
        
    def get_profile(self):
        if self.current_user != None:
            user = self.current_user
            
            print(f"\nName: {user}")
            print(f"Characters: {self.accounts[user]['characters']}")
            return True
        
        else:
            print("\nFirst, register or log in to your account")
            return False
        
    def create_character():
        pass


