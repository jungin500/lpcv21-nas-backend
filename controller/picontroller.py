import websockets
import asyncio
import uuid
import io


async def wprint(websocket, *args, **kwargs):
    print('>', *args, **kwargs)

    output = io.StringIO()
    print(*args, file=output, end='', **kwargs)
    contents = output.getvalue()
    output.close()
    await websocket.ping()
    await websocket.send(contents)


class PiControllerServer(object):
    """ Raspberry Pi Remote Controller Server over WebSocket

    Controls session between server(PC) and client(Raspberry Pi).
    server will have to control when/which model to infer,
    and send those information to Pi.

    When inference begin, server also should measure power usage of Pi
    and calculate average current usage of Pi.

    :Author:
        Jung-In An <ji5489@gmail.com>
        CVMIPLab, Kangwon National University
    """

    def __init__(self, listen_address='0.0.0.0', listen_port=12700):
        self.listen_address = listen_address
        self.listen_port = listen_port
        self.socket = None

        self.clients = []
        self.signal_buffer = []
        # assume instance exists after constructor
        PiControllerServer.Instance = self

    def __del__(self):
        PiControllerServer.Instance = None

    @staticmethod
    async def onmessage(w, path):
        self = PiControllerServer.Instance
        while True:
            message = await w.recv()
            if message.strip() == '':
                continue
            elif message.upper().startswith("CONNECT"):
                client_id = message.split(' ')[1:]
                await wprint(w, "Client %s connected!" % client_id)
                client_object = {
                    'client_id': client_id,
                    'websocket': w
                }
                self.clients.append(client_object)
                break
            else:
                await wprint(w, "Invalid handshake")
                return  # closes this session

        # Event Loop
        frame_total = frame_count = 0
        while True:
            message = await w.recv()
            if message.strip() == '':
                continue

            print("< %s" % message)
            message_slice = message.split(' ')
            command = message_slice[0].upper()
            args = message_slice[1:]

            self.signal_buffer.append(command)

            if command == 'DISCONNECT':
                self.clients.remove(client_object)
                await wprint(w, "Client %s disconnected!" % client_id)
                return
            elif command == 'BEGININFER':
                total_frame = int(args.pop())
                frame_total += total_frame
                await wprint(w, "Begin inference of model")
            elif command == 'ENDINFER':
                frame_elapsed_time = float(args.pop())
                await wprint(w, "End inference of model, time: %.4f" % frame_elapsed_time)
            elif command == 'NOMODEL':
                await wprint(w, "Client replied model can't be loaded! check client.")
            elif command == 'FRAME':
                frame_count += 1
                frame_elapsed_time = float(args.pop())
                await wprint(w, "Frame [%d/%d] Took %.4fms" % (frame_count, frame_total, frame_elapsed_time))
            else:
                await wprint(w, "Unknown command: %s, args:" % command, args)

    def serve(self):
        asyncio.get_event_loop().run_until_complete(self.serve_async())
        # asyncio.gather([
        #     asyncio.get_event_loop().run_until_complete(self.serve_async()),
        #     asyncio.get_event_loop().run_until_complete(self.wait_until_client_connects_async())
        # ])
        # asyncio.get_event_loop().run_forever()

    def bye(self):
        return asyncio.get_event_loop().run_until_complete(self.send_message_async("BYE"))

    def send_message(self, message):
        return asyncio.get_event_loop().run_until_complete(self.send_message_async(message))

    def send_binary(self, binary):
        return asyncio.get_event_loop().run_until_complete(self.send_binary_async(binary))

    def wait_until_client_connects(self):
        asyncio.get_event_loop().run_until_complete(self.wait_until_client_connects_async())

    def wait_until_signals(self, signal):
        return asyncio.get_event_loop().run_until_complete(self.wait_until_signals_async(signal))

    def get_last_signal(self):
        if len(self.signal_buffer) == 0:
            return None
        return self.signal_buffer[-1]

    async def serve_async(self):
        self.socket = await websockets.serve(self.onmessage, self.listen_address, self.listen_port, ping_interval=14400)

    async def send_message_async(self, message):
        return await self.clients[-1]['websocket'].send(message)

    async def send_binary_async(self, binary):
        return await self.clients[-1]['websocket'].send(binary, websockets.ABNF.OPCODE_BINARY)

    async def wait_until_client_connects_async(self):
        while True:
            if len(self.clients) > 0:
                break
            await asyncio.sleep(0.1)
        return True

    async def wait_until_signals_async(self, signal):
        signals = signal.split('|')
        while True:
            await asyncio.sleep(0.1)
            if len(self.signal_buffer) == 0:
                continue

            for signal in signals:
                if self.signal_buffer[-1] == signal:
                    value = self.signal_buffer.pop()
                    return value



class PiControllerClient(object):
    """ Raspberry Pi Remote Controller Client over WebSocket

    Client resides on Raspberry Pi.
    Pi is responsible of polling commands from server,
    load model and inference given model.

    :Author:
        Jung-In An <ji5489@gmail.com>
        CVMIPLab, Kangwon National University
    """

    def __init__(self, server_address: str):
        self.server_address = server_address
        self.socket = None

    def connect(self):
        return asyncio.get_event_loop().run_until_complete(self.connect_async())

    def send_message(self, message):
        return asyncio.get_event_loop().run_until_complete(self.send_message_async(message))

    def send_binary(self, binary):
        return asyncio.get_event_loop().run_until_complete(self.send_binary_async(binary))

    async def connect_async(self):
        self.socket = await websockets.connect(self.server_address, ping_interval=14400)
        await self.socket.send("CONNECT %s" % uuid.uuid1())
        return True

    async def send_message_async(self, message):
        await self.socket.ping()
        return await self.socket.send(message)

    async def send_binary_async(self, binary):
        await self.socket.ping()
        return await self.socket.send(binary, websockets.ABNF.OPCODE_BINARY)

    async def recv_message_async(self):
        return await self.socket.recv()
