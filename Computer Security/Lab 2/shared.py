import json
import bcrypt
from getpass import getpass


def get_data():
    try:
        file = open("baza.txt", "r", encoding='utf-8')
        data = json.loads(file.read())
        file.close()
        return data
    except:
        file = open("baza.txt", "x", encoding='utf-8')
        file.close()
        store_data(dict())
        return dict()


def store_data(data):
    file = open("baza.txt", "w", encoding='utf-8')
    file.write(json.dumps(data))
    file.close()


def check_password(password, hashed_password, salt):
    password = password.encode("utf-8")
    hashed_password_new = bcrypt.hashpw(password, salt)
    return hashed_password_new == hashed_password


def hash_password(password):
    password = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password, salt)
    return hashed_password, salt


def check_same(oldHash, oldSalt, newPass):
    return bcrypt.hashpw(newPass, oldSalt) == oldHash


def password_complexity_force():
    password = getpass("New password: ")
    while not minimum_complexity(password):
        print("Password must contain at least 8 characters, 1 lowercase letter, 1 uppercase letter and 1 number")
        password = getpass("New password: ")
    return password


def password_change(username):
        password = password_complexity_force()

        if password != getpass("Repeat new password: "):
            return False, "Password change failed. Password mismatch."
        
        data = get_data()
        if check_same(data[username]["hash"].encode("utf8"), data[username]["salt"].encode("utf-8"), password.encode("utf-8")):
            return False, "Password change failed. New password cannot be same as old."
        
        hashed_password, salt = hash_password(password)
        data[username] = {"hash" : hashed_password.decode("utf-8"), "salt" : salt.decode("utf-8"), "change" : False}
        store_data(data)
        return True, "Password change successful."


def minimum_complexity(password):
     return len(password) >= 8 \
        and any([letter.isupper() for letter in password]) \
        and any([letter.islower() for letter in password]) \
        and any([letter.isnumeric() for letter in password])
