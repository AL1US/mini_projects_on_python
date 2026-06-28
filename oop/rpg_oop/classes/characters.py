class Player:
    
    def __init__(self, name: str, player_class: str):
        self.user_name = name
        self.player_class = player_class
        self.HP = 0
        self.damage = 0
        self.magic_damage = 0
        self.armor = 0
        self.weapon = []
        self.backpack = []
        
    def get_stats(self):
        print(f"Name: {self.user_name}")
        print(f"Class: {self.player_class}")
        print(f"HP: {self.HP}")
        print(f"Damage: {self.damage}")
        print(f"magic_damage: {self.magic_damage}")
        print(f"armor: {self.armor}")

# Mage Class
class Mage(Player):
    def __init__(self, name: str):
        super().__init__(name, "Mage")
        
        self.user_class = "Mage"
        self.HP = 80
        self.damage = 3
        self.magic_damage = 15
        self.armor = 3
        self.weapon = ["Emerald Wand"]
        self.backpack = ["Emerald Wand","Power Potion"]
        
# Arrow Class
class Arrow(Player):
    
    def __init__(self, name: str):
        super().__init__(name, "Arrow")
        
        self.user_class = "Arrow"
        self.HP = 80
        self.damage = 10
        self.magic_damage = 10
        self.armor = 3
        self.weapon = ["Onion"]
        self.backpack = ["Onion", "Ordinary arrows","Magic Arrows"]

# Tank Class
class Tank(Player):
    def __init__(self, name: str):
        super().__init__(name, "Tank")
        
        self.user_class = "Tank"
        self.HP = 200
        self.damage = 5
        self.magic_damage = 5
        self.armor = 15
        self.weapon = ["Shield"]
        self.backpack = ["Shield","A Health Potion"] 

# Warrior Class
class Warrior(Player):
    
    def __init__(self, name: str):
        
        super().__init__(name, "Warrior")
        
        self.user_class = "Warrior"
        self.HP = 150
        self.damage = 15
        self.magic_damage = 0
        self.armor = 10
        self.weapon = ["Sword"]
        self.backpack = ["Sword", "Beer"]
        