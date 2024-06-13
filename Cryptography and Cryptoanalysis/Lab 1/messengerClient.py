#!/usr/bin/env python3
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

class Connection:
    def __init__(self, chain_key_send, chain_key_recv, max_skip):
        self.chain_key_send = chain_key_send
        self.chain_key_recv = chain_key_recv

        self.max_skip = max_skip
        self.skipped = {} # key = N, val = msg_key
        self.N_recv = 0
        self.N_send = 0

        self.msg_key_send = None
        self.msg_key_recv = None

    def kdf(self, chain_key):
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=64,
            salt=None,
            info=None,
        )
        new_key = hkdf.derive(chain_key)
        return new_key[0:32], new_key[32:]

    def ratchet_step_send(self):
        self.chain_key_send, self.msg_key_send = self.kdf(self.chain_key_send)
        self.N_send += 1

    def get_msg_key(self, N):
        if N > self.N_recv:
            while N > self.N_recv:
                self.chain_key_recv, self.msg_key_recv = self.kdf(self.chain_key_recv)
                self.N_recv += 1
                self.skipped[self.N_recv] = self.msg_key_recv
        
        if N not in self.skipped:
            raise Exception("Replay attack detected")
        key = self.skipped[N]
        self.skipped.pop(N)

        return key

    

class Message:
    def __init__(self, N, iv, ciphertext):
        self.N = N
        self.iv = iv
        self.ciphertext = ciphertext


class MessengerClient:
    """ Messenger client class

        Feel free to modify the attributes and add new ones as you
        see fit.

    """

    def __init__(self, username, max_skip=10):
        """ Initializes a client

        Arguments:
        username (str) -- client name
        max_skip (int) -- Maximum number of message keys that can be skipped in
                          a single chain

        """
        self.username = username
        # Data regarding active connections.
        self.conn = {}
        # Maximum number of message keys that can be skipped in a single chain
        self.max_skip = max_skip


    def add_connection(self, username, chain_key_send, chain_key_recv):
        """ Add a new connection

        Arguments:
        username (str) -- user that we want to talk to
        chain_key_send -- sending chain key (CKs) of the username
        chain_key_recv -- receiving chain key (CKr) of the username

        """

        self.conn[username] = Connection(chain_key_send, chain_key_recv, self.max_skip)

        


    def send_message(self, username, message):
        """ Send a message to a user

        Get the current sending key of the username, perform a symmetric-ratchet
        step, encrypt the message, update the sending key, return a header and
        a ciphertext.

        Arguments:
        username (str) -- user we want to send a message to
        message (str)  -- plaintext we want to send

        Returns a ciphertext and a header data (you can use a tuple object)

        """
        if username not in self.conn:
            raise Exception("No such connection")
        
        connection = self.conn[username]
        connection.ratchet_step_send()
        iv = os.urandom(12)
        ciphertext = self.encrypt(message, connection.msg_key_send, iv)
        msg = Message(connection.N_send, iv, ciphertext)

        #print("\n", self.username, message, connection.N_send, iv, ciphertext, connection.msg_key_send)

        return msg


    def receive_message(self, username, message):
        """ Receive a message from a user

        Get the username connection data, check if the message is out-of-order,
        perform necessary symmetric-ratchet steps, decrypt the message and
        return the plaintext.

        Arguments:
        username (str) -- user who sent the message
        message        -- a ciphertext and a header data

        Returns a plaintext (str)

        """
        if username not in self.conn:
            raise Exception("No such connection")
        
        connection = self.conn[username]
        N = message.N
        iv = message.iv
        ciphertext = message.ciphertext

        if N - connection.N_recv > connection.max_skip:
            raise Exception("Skip limit exceeded")
        
        key = connection.get_msg_key(N)

        msg = self.decrypt(ciphertext, key, iv)

        return msg

    def encrypt(self, plaintext, key, iv):
        plaintext = plaintext.encode()

        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(iv, plaintext, None)
        return ct

    def decrypt(self, ciphertext, key, iv):
        aesgcm = AESGCM(key)
        pt = aesgcm.decrypt(iv, ciphertext, None)

        pt = pt.decode()
        return pt