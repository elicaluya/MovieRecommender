from User import User
import pandas as pd


class UserData:
    def __init__(self, df_user = None):
        if not df_user:
            # loads user dataset files
            self.df_user = pd.read_csv(
                "movielens/Movielens-02/u.user",
                delimiter="|",
                names=['user_id', 'age', 'gender', 'occupation', 'zipcode'])
        else:
            self.df_user = df_user

    def get_max_id(self):
        return self.df_user["user_id"].max()
        
    def get_user(self, user_id):
        """Finds specific user in the user dataset by given id.

        Args:
            user_id: user id (not index of the table)

        Returns:
            A User object containing attributes from the user dataset.
        """
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
        self.user_data = user_data
        if not df_app_user:
            # loads app user dataset files
            try:
                self.df_app_user = pd.read_csv("app_user.csv")
            except FileNotFoundError as e:
                # start with empty set
                self.df_app_user = pd.DataFrame(
                    columns=['user_id', 'name', 'age', 'gender'])
        else:
            self.df_app_user = df_app_user

        self.max_app_user_id = self.df_app_user["user_id"].max()

    def get_user(self, user_id):
        """Finds specific user in the app user dataset by given id.

        Args:
            user_id: user id (not index of the table)

        Returns:
            A User object containing attributes from the app user dataset.
        """
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
        max_user_id = self.user_data.get_max_id()
        new_user_id = max(max_user_id, self.max_app_user_id) + 1
        new_row = {'user_id': new_user_id, 'name': name, 'age': age, 'gender': gender}
        self.df_app_user = self.df_app_user.append(new_row, ignore_index=True)
        self.max_app_user_id = new_user_id
        return User(new_user_id, name, age, gender)

    def save(self):
        """Store all the app user into app user files."""
        self.df_app_user = self.df_app_user.to_csv("app_user.csv", index=False)