import json
import queue
import asyncio
import threading
import websockets

DEBUG_MODE = True

IP = '0.0.0.0'
PORT = 14285
ROOM_ID = '1'
SEAT = '0'

#pylint: disable = R0201
#pylint: disable = R0801
#pylint: disable = E1111
#pylint: disable = W0612
#pylint: disable = W0105
#pylint: disable = W0107


def rserver_convert_byte(data_str):
    '''
    传输数据的时候加数据长度作为数据头
    '''
    message_len = len(data_str)
    message = message_len.to_bytes(4, byteorder='big', signed=True)
    message += bytes(data_str, encoding="utf8")
    return message


class RServer():
    '''
    跟播放器建立websocket连接
    '''
    USERS = set()
    player = None

    recv_msg_queue = queue.Queue()
    send_msg_queue = queue.Queue()

    server_thread = threading.Thread

    def __init__(self, ip=IP, port=PORT, room_id=ROOM_ID, seat=SEAT, loop=None):
        '''
        初始化函数
        '''
        self.seat = 0
        self.judger = None
        self.std_thread = None
        self.ip = ip
        self.port = port
        self.room_id = room_id
        self.seat = seat
        self.loop = loop
        self.end_tag = False
        self.type_tag = 2
        self.length_limit = 2 ** 16
        self.server = websockets.serve

    def build_server(self):
        '''
        准备建立连接
        '''
        self.server_thread = threading.Thread(target=self.run)
        self.server_thread.start()

    def destory(self):
        '''
        结束协程
        '''
        # coro = self.player.close()
        # future = asyncio.ensure_future(coro, self.loop)
        # self.server.close()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.server_thread.join()
        if self.judger is not None:
            self.judger.awake()

    async def register(self, websocket):
        '''
        register
        '''
        self.USERS.add(websocket)

    async def unregister(self, websocket):
        '''
        unregister
        '''
        self.USERS.remove(websocket)

    def player_connect(self, ws, token: str):
        '''
        token = base64.b64decode(token).decode()
        judge_ip, lst = token.split(':')
        port, room, username, seat = token.split('/')
        if '' in [judge_ip, port, room, username, seat]:
            logger.info('token %s invalid' % token)
            return False
        '''
        if self.player is not None:
            return False

        self.player = ws
        self.std_thread.write(rserver_convert_byte(json.dumps({'type': 1, 'index': int(self.seat), 'success': 1})))

        return True

    async def handler(self, websocket, path):
        '''
        协程
        '''
        try:
            await self.register(websocket)
            async for message in websocket:
                try:
                    data = json.loads(message)
                    token = data["token"]
                    if data["request"] == "connect":
                        flag = self.player_connect(websocket, token)
                        if not flag:
                            break
                    elif data["request"] == "action":
                        self.recv_msg(data["content"])
                except:
                    pass
                    #logger.exception('error occured when reading message from %s' % websocket)
        except:
            pass
            #logger.exception('in rserver handler')
        finally:
            await self.unregister(websocket)

    def run(self):
        '''
        启动
        '''
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            start_server = websockets.serve(self.handler, self.ip, self.port)
            self.server = self.loop.run_until_complete(start_server)
            self.loop.run_forever()
        except:
            pass
            #logging.exception('invalid at run')

    def join(self):
        '''
        合并接口
        '''
        pass

    def recv_msg(self, msg):
        '''
        接收消息
        '''
        #logger.info('seat %s received msg:\n%s' % (self.seat, msg))
        try:
            self.judger.receive_message(msg, int(self.seat))
        except:
            pass
            #logger.exception('in receiving message')

    def get_msg(self):
        '''
        加入消息队列
        '''
        while self.recv_msg_queue.empty():
            continue
        return self.recv_msg_queue.get()

    async def test_send_msg(self, msg):
        '''
        尝试发送消息
        '''
        await self.player.send(msg)
        #logger.info('sending message complete')

    def send_msg(self, msg):
        '''
        发送消息
        '''
        try:
            #logger.info('ensuring future with message:\n%s' % msg)
            #asyncio.ensure_future(self.test_send_msg(msg), loop=self.loop)
            coro = self.test_send_msg(msg)
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        except:
            pass
            #logger.exception('in rserver sending message')

    def send2player(self, msg, request='action'):
        '''
        发送给玩家消息
        '''
        self.send_msg(json.dumps({'request': request, 'content': msg}))

    def set_judger(self, judger):
        '''
        设置judger
        '''
        self.judger = judger

    def set_std(self, std_thread):
        '''
        设置标准输入输出线程
        '''
        self.std_thread = std_thread

    def write(self, msg):
        '''
        写函数
        '''
        #logger.info('writing message to remote player')
        self.send2player(msg.decode())

    def change_length(self, length):
        '''
        改变接收消息的长度
        '''
        del length

    def start(self):
        '''
        启动函数
        '''
        return None
