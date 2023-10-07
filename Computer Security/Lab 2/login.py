import typer
import shared
from getpass import getpass

app = typer.Typer()

@app.command()
def login(username: str):
    password = getpass()
    data = shared.get_data()
    
    if not username in data.keys() or not shared.check_password(password, data[username]["hash"].encode("utf-8"), data[username]["salt"].encode("utf-8")):
        print("Username or password incorrect.")
        return
    elif data[username]["change"]:
        rez, text = shared.password_change(username)
        if rez:
            data[username]["try"] = 0
            shared.store_data(data)
        print(text)
    else:
        print("Login sucessful.")

app()