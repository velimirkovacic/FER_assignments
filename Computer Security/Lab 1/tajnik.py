
import typer
import json
from base64 import b64encode, b64decode
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import HMAC, SHA512
from Crypto.Random import get_random_bytes
from Crypto.Cipher import ChaCha20


def get_data():
    file = open("baza.txt", "r", encoding='utf-8')
    data = json.loads(file.read())
    file.close()
    data["nonce"] = b64decode(data["nonce"])
    data["salt"] = b64decode(data["salt"])
    return data

def store_data(data):
    file = open("baza.txt", "w", encoding='utf-8')
    data["data"] = b64encode(data["data"]).decode("utf-8")
    data["nonce"] = b64encode(data["nonce"]).decode('utf-8')
    data["salt"] = b64encode(data["salt"]).decode('utf-8')
    file.write(json.dumps(data))
    file.close()


def key_derivation(masterPassword, database, salt=None):
    if salt == None:
        salt = get_random_bytes(16)

    keys = PBKDF2(masterPassword, salt, 160, count=1000000, hmac_hash_module=SHA512)
    database["salt"] = salt
    return keys[:128], keys[128:]


def generate_mac(mac_key, cipher, nonce, salt):
    h = HMAC.new(mac_key, digestmod=SHA512)
    h.update(cipher)
    h.update(nonce)
    h.update(salt)

    return h.hexdigest()


def verify_mac(mac, mac_key, cipher, nonce, salt):
    h = HMAC.new(mac_key, digestmod=SHA512)
    h.update(cipher)
    h.update(nonce)
    h.update(salt)

    h.hexverify(mac)



def encrypt(database, key):
    nonce = get_random_bytes(12)
    cipher = ChaCha20.new(key=key, nonce=nonce)
    ciphertext = cipher.encrypt(str.encode(database["data"]))

    database["data"] = ciphertext
    database["nonce"] = nonce


def decrypt(database, key):
    nonce = database["nonce"]
    ciphertext = b64decode(database["data"])
    cipher = ChaCha20.new(key = key, nonce = nonce)
    plaintext = cipher.decrypt(ciphertext)
    return json.loads(plaintext)



app = typer.Typer()


@app.command()
def init(masterpassword: str):
    file = open("baza.txt", "w")
    file.write(json.dumps({"data": "", "nonce": "", "salt" : "", "mac": ""}))
    file.close()

    database = get_data()
    database["data"] = json.dumps(dict())

    mac_key, symmetric_key = key_derivation(masterpassword, database)

    encrypt(database, symmetric_key)

    mac = generate_mac(mac_key, bytearray(b64encode(database["data"]).decode('utf-8'), "utf-8"), bytearray(b64encode(database["nonce"]).decode('utf-8'), "utf-8"), bytearray(b64encode(database["salt"]).decode('utf-8'), "utf-8"))
    database["mac"] = mac
    store_data(database)
    print("Password manager initialized.")


@app.command()
def put(masterpassword: str, address: str, password: str):

    try:
        database = get_data()

        mac_key, symmetric_key = key_derivation(masterpassword, database, database["salt"])
        verify_mac(database["mac"], mac_key, bytearray(database["data"], "utf-8"), bytearray(b64encode(database["nonce"]).decode('utf-8'), "utf-8"), bytearray(b64encode(database["salt"]).decode('utf-8'), "utf-8"))
        adr_pass = decrypt(database, symmetric_key)
    except:
        print("Master password incorrect or integrity check failed.")
        return

    adr_pass[address] = password
    database["data"] = json.dumps(adr_pass)

    encrypt(database, symmetric_key)
    mac = generate_mac(mac_key, bytearray(b64encode(database["data"]).decode('utf-8'), "utf-8"), bytearray(b64encode(database["nonce"]).decode('utf-8'), "utf-8"), bytearray(b64encode(database["salt"]).decode('utf-8'), "utf-8"))
    database["mac"] = mac
    store_data(database)

    print("Stored password for: {}.".format(address))


@app.command()
def get(masterpassword: str, address: str):
    try:
        database = get_data()

        mac_key, symmetric_key = key_derivation(masterpassword, database, database["salt"])
        verify_mac(database["mac"], mac_key, bytearray(database["data"], "utf-8"), bytearray(b64encode(database["nonce"]).decode('utf-8'), "utf-8"), bytearray(b64encode(database["salt"]).decode('utf-8'), "utf-8"))
        adr_pass = decrypt(database, symmetric_key)
    except:
        print("Master password incorrect or integrity check failed.")
        return
    if address not in adr_pass.keys():
        print("There is no password stored for: {}.".format(address))
    else:
        print("Password for {} is: {}.".format(address, adr_pass[address]))


app()