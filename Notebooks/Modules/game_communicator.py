import socket

class GameCommunicator:
    server_port = 5000

    def __init__(self):

        """
        Constructor used to create a new connection with the client (game client).
        """
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create the socket
        self.server_socket.bind(('localhost', self.server_port))  # Bind the socket with the port
        self.server_socket.listen(1)



    def wait_for_connection(self):

        """
        Function used to wait for an incoming connection.
        """

        self.client_connection, self.client_address = self.server_socket.accept()  # Wait for a game connection
        print('Connection from', self.client_address)



    def send_to_game(self, msg, verbose=True):

        """
        Function used to send a message to the game.

        Parameters:
        - msg: The message to be sent to the game.
        """

        try:
            message = msg.encode()  # Encode the message
            self.client_connection.sendall(message)  # Send the message using the socket
            if(verbose):
                print(f'Sent to game: {msg}')
        except Exception as e:
            print(f"Error sending message to game: {e}")



    def receive_from_game(self):

        """
        Function used to receive a message from the game.

        Returns:
        - The received message from the game.
        """

        while True:
            try:
                data = self.client_connection.recv(1024)  # Wait for data from the game
                if not data:
                    continue  # No data received, continue waiting
                received_msg = data.decode()  # Decode the data
                print(f'Received from game: {received_msg}')
                return received_msg
            except Exception as e:
                print(f"Error receiving message from game: {e}")
                return None



    def close_connection(self):

        """
        Function used to close the connection.
        """

        try:
            self.client_connection.close()  # Close the connection
            self.server_socket.close()  # Close the socket
            print("Connection closed")
        except Exception as e:
            print(f"Error closing connection: {e}")
