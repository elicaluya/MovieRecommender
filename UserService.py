"""User service layer object module.

This module includes a class representing service related with user data.
It links to data layer object to deliver the functionality.
"""

class UserService:
    def __init__(self, user_data, app_user_data):
        """Constructor, it connects data objects in lower level operation layer.

        Args:
            user_data: base user dataset module
            app_user_data: app user dataset module
        """
        self.user_data = user_data
        self.app_user_data = app_user_data

    def __search_app_user(self, user_id):
        """Finds specific user in the app user dataset by given id.

        Args:
            user_id: user id (not index of the table)

        Returns:
            A User object containing attributes from the app user dataset.
        """
        return self.app_user_data.get_user(user_id)
    
    def __search_user(self, user_id):
        """Finds specific user in the user dataset by given id.

        Args:
            user_id: user id (not index of the table)

        Returns:
            A User object containing attributes from the user dataset.
        """
        return self.user_data.get_user(user_id)
    
    def search_user(self, user_id):
        """Finds specific user in both user dataset (movielens and app) by given id.
        It delegates actual searching down to two relevant search functions.

        Args:
            user_id: user id (not index of the table), this is unique in both dataseet.

        Returns:
            A User object containing attributes from the either user dataset.
        """
        user = self.__search_app_user(user_id)
        if user:
            return user
        # then search in user list
        return self.__search_user(user_id)

    def create_new_user(self, name, age, gender):
        """Create an app user by given parameters.

        Args:
            name: name string
            age: age in positive integer
            gender: M or F

        Returns:
            A User object containing attributes generated from given parameters.
        """
        return self.app_user_data.add_user(name, age, gender)

    def save_app_users(self):
        """Store all the app user into app user files."""
        self.app_user_data.save()
