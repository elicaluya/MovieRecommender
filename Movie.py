"""Global structure for movie object shared by all the components. 
It represents a simple container object for movie attributes."""
class Movie:
    def __init__(self, id, title, genre):
        """Constructor for a Movie object by given attributes.

        Args:
            id: movie id
            title: movie title in full length
            genre: movie genre
        """
        self.id = id
        self.title = title
        self.genre = genre
    
    # getter methods
    def get_id(self):
        return self.id
    
    def get_title(self):
        return self.title
    
    def get_genre(self):
        return self.genre