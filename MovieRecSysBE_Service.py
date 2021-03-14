class RecommendService:
    def __init__(self):
        return

    def recommend_movie_rating(self, movie_id, user_id):
        return

    def recommend_movie_by_genre(self, genres):
        return

class UserService:
    def __init__(self, user_service):
        self.user_service = user_service

    def create_new_user(self, name, age, gender):
        """Create an app user by given parameters.

        Args:
            name: name string
            age: age in positive integer
            gender: M or F

        Returns:
            A User object containing attributes generated from given parameters.
        """
        return self.user_service.create_new_user(name, age, gender)

    def find_user(self, user_id):
        """Finds specific user by given user id.

        Args:
            user_id: unique user id.

        Returns:
            A User object containing attributes.
        """
        return self.user_service.search_user(user_id)

    def change_user(self, user_id):
        # Not implemented yet
        raise NotImplementedError("Not Implemented yet")