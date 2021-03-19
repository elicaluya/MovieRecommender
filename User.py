"""Global structure for user object shared by all the components. 
It represents a simple container object for user attributes as of now. 
But possibly in a bigger system, it may present full sized message structure."""

class User:
    def __init__(self, id, name, age, gender):
        """Constructor for a User object by given attributes.

        Args:
            id: user id
            name: name of the user
            age: age in number
            gender: M or F for gender
        """
        self.id = int(id)
        self.name = name
        self.age = int(age)
        self.gender = gender
        self.ratings = None
    
    # getter methods
    def get_name(self):
        return self.name
    
    def get_age(self):
        return self.age
    
    def get_gender(self):
        return self.gender
    
    def get_id(self):
        return self.id
