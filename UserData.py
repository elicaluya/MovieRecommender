"""User data layer object module.

This module provides low level access to data layer user dataset handling.
"""
from User import User
import pandas as pd


class UserData:
    def __init__(self, df_user = None):
        """Constructor, reads dataset files or use given dataset for rating and matrix information.

        Args:
            df_user: custom user information dataset
        """
        # base user information dataset
        if df_user is None:
            # use base dataset read from file
            self.df_user = pd.read_csv(
                "movielens/Movielens-02/u.user",
                delimiter="|",
                names=['user_id', 'age', 'gender', 'occupation', 'zipcode'])
        else:
            # use given custom dataset
            self.df_user = df_user

    def get_max_id(self):
        # return maximum id in base dataset, which in turn used for app user id base
        return self.df_user["user_id"].max()
        
    def get_user(self, user_id):
        """Finds specific user in the user dataset by given id.

        Args:
            user_id: user id (not index of the table)

        Returns:
            A User object containing attributes from the user dataset.
        """
        # user not found if base dataset is empty
        if self.df_user.empty:
            return None
        found_df = self.df_user.loc[self.df_user["user_id"] == int(user_id)]
        if found_df.empty:
            return None
        row = found_df.iloc[0]
        print("DEBUG: found user:", user_id, "", row["age"],
              row["gender"], row["occupation"], row["zipcode"])
        return User(user_id, "", row["age"], row["gender"])


class AppUserData:
    def __init__(self, user_data, df_app_user = None):
        """Constructor, reads dataset files or use given dataset for app user information.

        Args:
            user_data: it requires access to the other base user object
            df_app_user: custom app user information dataset
        """
        self.user_data = user_data
        # app user information dataset
        if df_app_user is None:
            # use base dataset read from file
            try:
                self.df_app_user = pd.read_csv("app_user.csv")
            except FileNotFoundError as e:
                # start with empty set, if not found
                self.df_app_user = pd.DataFrame(
                    columns=['user_id', 'name', 'age', 'gender'])
        else:
            # use given custom dataset
            self.df_app_user = df_app_user

        # this is for user identification whether it belongs to app or base set
        self.max_app_user_id = self.df_app_user["user_id"].max()

    def get_user(self, user_id):
        """Finds specific user in the app user dataset by given id.

        Args:
            user_id: user id (not index of the table)

        Returns:
            A User object containing attributes from the app user dataset.
        """
        # user not found if base dataset is empty
        if self.df_app_user.empty:
            return None
        found_df = self.df_app_user.loc[self.df_app_user["user_id"] == int(user_id)]
        if found_df.empty:
            return None
        row = found_df.iloc[0]
        print("DEBUG: found app user:", user_id, row["name"], row["age"], row["gender"])
        return User(user_id, row["name"], row["age"], row["gender"])

    def add_user(self, name, age, gender):
        """Create an app user by given parameters.

        Args:
            name: name string
            age: age in positive integer
            gender: M or F

        Returns:
            A User object containing attributes generated from given parameters.
        """
        # new user id based on max id from either dataset in order to keep uniqueness
        max_user_id = self.user_data.get_max_id()
        new_user_id = max(max_user_id, self.max_app_user_id) + 1
        new_row = {'user_id': new_user_id, 'name': name, 'age': age, 'gender': gender}
        self.df_app_user = self.df_app_user.append(new_row, ignore_index=True)
        self.max_app_user_id = new_user_id
        return User(new_user_id, name, age, gender)

    def save(self):
        """Store all the app user into app user files."""
        self.df_app_user = self.df_app_user.to_csv("app_user.csv", index=False)