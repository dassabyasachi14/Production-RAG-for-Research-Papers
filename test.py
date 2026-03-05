class Dog:
    def __init__(self,name):
        self.name=name
    
    def bark(self):
        return f"{self.name} is barking"

#jerry = Dog("jerry")


class Trained_Dog(Dog):
    def sniffing(self):
        return f"{self.name} is sniffing"

jerry=Trained_Dog("Jerry")

jerry.sniffing()

