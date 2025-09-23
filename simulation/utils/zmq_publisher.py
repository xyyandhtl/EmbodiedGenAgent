import zmq
import pickle

class ZMQDataPublisher:
    """
    A simple ZeroMQ publisher to send Python objects from the simulation.
    """
    def __init__(self, port=5555):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        print(f"[ZMQ Publisher] Started on port {port}. Waiting for subscribers...")

    def publish_data(self, data):
        """
        Serializes and sends the data dictionary.
        """
        try:
            # Serialize the data using pickle
            message = pickle.dumps(data)
            self.socket.send(message)
        except Exception as e:
            print(f"[ZMQ Publisher] Error sending data: {e}")

    def close(self):
        self.socket.close()
