import typer
import shared
from getpass import getpass

app = typer.Typer()

@app.command()
def add(username: str):
    password = shared.password_complexity_force()
        
    if password != getpass("Repeat Password: "):
        print("User add failed. Password mismatch.")
    else:
        data = shared.get_data()
        hashed_password, salt = shared.hash_password(password)
        data[username] = {"hash" : hashed_password.decode("utf-8"), "salt" : salt.decode("utf-8"), "change" : False}
        shared.store_data(data)
        print("User", username, "successfuly added.")


@app.command()
def passwd(username: str):
    data = shared.get_data()
    if username not in data.keys():
        print("User does not exist.")
        return

    print(shared.password_change(username)[1])


@app.command()
def forcepass(username: str):
    data = shared.get_data()
    if username in data.keys():
        data[username]["change"] = True
        shared.store_data(data)
        print("User will be requested to change password on next login.")
    else:
        print("User does not exist.")


@app.command()
def dlt(username: str):
    data = shared.get_data()
    if username in data.keys():
        data.pop(username)
        print("User successfuly removed.")
    else:
        print("User does not exist.")

app()