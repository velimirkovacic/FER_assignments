#!/usr/bin/env python3

import pickle
import os
import pyDH
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class Message:
    def __init__(self, public_dh, N, iv, ciphertext):
        self.public_dh = public_dh
        self.N = N
        self.iv = iv
        self.ciphertext = ciphertext


class Connection:
    def __init__(self, dh_key_pair, other_public_dh):
        self.dh_key_pair = dh_key_pair
        self.other_public_dh = other_public_dh

        self.RK = None
        self.DHs, self.DHr = None, None
        self.CKs, self.CKr = None, None
        
        self.Ns = 0
        self.Nr = 0
        #self.PN = 0

    def init_recv(self, m):
        self.DHs = self.dh_key_pair
        self.RK = str(self.dh_key_pair.gen_shared_key(self.other_public_dh)).encode()
        self.DH_ratchet(m)

    def init_send(self):
        self.DHs = self.dh_key_pair
        self.DHr = self.other_public_dh
        self.RK = str(self.dh_key_pair.gen_shared_key(self.DHr)).encode()
        self.RK, self.CKs = self.kdf(self.RK, str(self.dh_key_pair.gen_shared_key(self.DHr)).encode())

    def generate_dh(self):
        return pyDH.DiffieHellman()
    
    def kdf(self, chain_key, info=None):
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=64,
            salt=None,
            info=info,
        )
        new_key = hkdf.derive(chain_key)
        return new_key[0:32], new_key[32:]

    def DH_ratchet(self, m):
        self.PN = self.Ns                          
        self.Ns = 0
        self.Nr = 0
        self.DHr = m.public_dh
        self.RK, self.CKr = self.kdf(self.RK, str(self.DHs.gen_shared_key(self.DHr)).encode())
        self.DHs = self.generate_dh()
        self.RK, self.CKs = self.kdf(self.RK, str(self.DHs.gen_shared_key(self.DHr)).encode())


    def encrypt(self, plaintext):
        if self.DHs == None: 
            self.init_send() # Incijalizacija u slučaju da prvi šalje

        self.CKs, mk = self.kdf(self.CKs)
        plaintext = plaintext.encode()
        aesgcm = AESGCM(mk)
        iv = os.urandom(12)
        ciphertext = aesgcm.encrypt(iv, plaintext, None)

        m = Message(self.DHs.gen_public_key(), self.Ns, iv, ciphertext)
        self.Ns += 1

        return m
    
    def decrypt(self, m): 
        if self.DHs == None: # Incijalizacija u slučaju da prvi prima
            self.init_recv(m)

        if m.public_dh != self.DHr:
            self.DH_ratchet(m)
        mk = None

        while m.N > self.Nr:
            self.CKr, mk = self.kdf(self.CKr)
            self.Nr += 1
        self.CKr, mk = self.kdf(self.CKr)
        self.Nr += 1

        aesgcm = AESGCM(mk)
        pt = aesgcm.decrypt(m.iv, m.ciphertext, None)
        pt = pt.decode()

        return pt


class MessengerClient:
    """ Messenger client klasa

        Slobodno mijenjajte postojeće atribute i dodajte nove kako smatrate
        prikladnim.
    """

    def __init__(self, username, ca_pub_key):
        """ Inicijalizacija klijenta

        Argumenti:
        username (str) -- ime klijenta
        ca_pub_key     -- javni ključ od CA (certificate authority)

        """
        self.username = username
        self.ca_pub_key = ca_pub_key
        # Aktivne konekcije s drugim klijentima
        self.conns = {}
        # Inicijalni Diffie-Hellman par ključeva iz metode `generate_certificate`
        self.dh_key_pair = ()

    

    def generate_certificate(self):
        """ Generira par Diffie-Hellman ključeva i vraća certifikacijski objekt

        Metoda generira inicijalni Diffie-Hellman par kljuceva; serijalizirani
        javni kljuc se zajedno s imenom klijenta postavlja u certifikacijski
        objekt kojeg metoda vraća. Certifikacijski objekt moze biti proizvoljan (npr.
        dict ili tuple). Za serijalizaciju kljuca mozete koristiti
        metodu `public_bytes`; format (PEM ili DER) je proizvoljan.

        Certifikacijski objekt koji metoda vrati bit će potpisan od strane CA te
        će tako dobiveni certifikat biti proslijeđen drugim klijentima.

        """
        dh = pyDH.DiffieHellman()
        dh_pubkey = dh.gen_public_key()
        self.dh_key_pair = dh

        return (dh_pubkey, self.username)
         

    def receive_certificate(self, cert, signature):
        """ Verificira certifikat klijenta i sprema informacije o klijentu (ime
            i javni ključ)

        Argumenti:
        cert      -- certifikacijski objekt
        signature -- digitalni potpis od `cert`

        Metoda prima certifikacijski objekt (koji sadrži inicijalni
        Diffie-Hellman javni ključ i ime klijenta) i njegov potpis kojeg
        verificira koristeći javni ključ od CA i, ako je verifikacija uspješna,
        sprema informacije o klijentu (ime i javni ključ). Javni ključ od CA je
        spremljen prilikom inicijalizacije objekta.
        """
        try:
            self.ca_pub_key.verify(signature, pickle.dumps(cert), ec.ECDSA(hashes.SHA256()))
            self.conns[cert[1]] = Connection(self.dh_key_pair, cert[0])

        except(InvalidSignature):
            pass
        

    def send_message(self, username, message):
        """ Slanje poruke klijentu

        Argumenti:
        message  -- poruka koju ćemo poslati
        username -- klijent kojem šaljemo poruku `message`

        Metoda šalje kriptiranu poruku sa zaglavljem klijentu s imenom `username`.
        Pretpostavite da već posjedujete certifikacijski objekt od klijenta
        (dobiven pomoću `receive_certificate`) i da klijent posjeduje vaš.
        Ako već prije niste komunicirali, uspostavite sesiju tako da generirate
        nužne `double ratchet` ključeve prema specifikaciji.

        Svaki put kada šaljete poruku napravite `ratchet` korak u `sending`
        lanacu (i `root` lanacu ako je potrebno prema specifikaciji).  S novim
        `sending` ključem kriptirajte poruku koristeći simetrični kriptosustav
        AES-GCM tako da zaglavlje poruke bude autentificirano.  Ovo znači da u
        zaglavlju poruke trebate proslijediti odgovarajući inicijalizacijski
        vektor.  Zaglavlje treba sadržavati podatke potrebne klijentu da
        derivira novi ključ i dekriptira poruku.  Svaka poruka mora biti
        kriptirana novim `sending` ključem.

        Metoda treba vratiti kriptiranu poruku zajedno sa zaglavljem.

        """

        if username not in self.conns.keys():
            raise KeyError

        return self.conns[username].encrypt(message)

    def receive_message(self, username, message):
        """ Primanje poruke od korisnika

        Argumenti:
        message  -- poruka koju smo primili
        username -- klijent koji je poslao poruku

        Metoda prima kriptiranu poruku od klijenta s imenom `username`.
        Pretpostavite da već posjedujete certifikacijski objekt od klijenta
        (dobiven pomoću `receive_certificate`) i da je klijent izračunao
        inicijalni `root` ključ uz pomoć javnog Diffie-Hellman ključa iz vašeg
        certifikata.  Ako već prije niste komunicirali, uspostavite sesiju tako
        da generirate nužne `double ratchet` ključeve prema specifikaciji.

        Svaki put kada primite poruku napravite `ratchet` korak u `receiving`
        lanacu (i `root` lanacu ako je potrebno prema specifikaciji) koristeći
        informacije dostupne u zaglavlju i dekriptirajte poruku uz pomoć novog
        `receiving` ključa. Ako detektirate da je integritet poruke narušen,
        zaustavite izvršavanje programa i generirajte iznimku.

        Metoda treba vratiti dekriptiranu poruku.

        """
        if username not in self.conns.keys():
            raise KeyError

        return self.conns[username].decrypt(message)

def main():
    pass

if __name__ == "__main__":
    main()
