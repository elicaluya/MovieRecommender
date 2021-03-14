class UserService:
    def __init__(self, user_data, app_user_data):
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
