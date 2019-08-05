import numpy as np
from PIL import Image

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import cv2

import time

import socket
import subprocess
import os




#### POSSO TENTAR CRIAR UM GYM QUE É APENAS UM HANDLE PARA A COMUNICACAO POR SOCKET COM O JOGO E USAR QUALQUER CÓDIGO RL COM BASE NESSE GYM 




class RealHalite(gym.Env):


    simulador = []
    map_size = (6,6)
    #reward = 0

    @property
    def action_space(self):
        return spaces.Discrete(5)
        #return spaces.Box(low=-1, high=1, shape=(1,)) # quando trabalhar com acoes continuas, vai ficar mais ou menos assim
        '''
        if(self.simulador.soloplay):
            return spaces.Discrete(5)
        if(self.simulador.duoplay):
            return spaces.Discrete(10)
        '''

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1000, shape=(self.map_size[0], self.map_size[1], 4))
        '''
        if(self.simulador.soloplay):
            return spaces.Box(low=0, high=1000, shape=(self.map_size[0], self.map_size[1], 4)) # quando trabalhar com imagem, vai ficar mais ou menos assim

        if(self.simulador.duoplay):
            return spaces.Box(low=0, high=1000, shape=(self.map_size[0], self.map_size[1], 5)) # quando trabalhar com imagem, vai ficar mais ou menos assim
            '''
        #return spaces.Box(low=np.array([-3.1415,-20.0, -inf, -inf]), high=np.array([3.1415,20.0, inf, inf])) # 4 estados, com boundaries dados ([ang, erro linear, vel ang, vel linear])

    def __init__(self):
        self.start_socket()

        '''
        self.simulador = HaliteSimulator.HaliteGame()
        self.player_halite_prev = 0
        '''
        #self.prevstate = deepcopy(self.simulador.state)



    def setupSocket(self):
        if(hasattr(self,'s')):
            self.s.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         # Create a socket object
        s.settimeout(5)
        #host = socket.gethostname() # Get local machine name
        host = ''
        port = 7080                # Reserve a port for your service.

        print('Server started!')
        print('Waiting for clients...')
        s.bind((host, port))        # Bind to the port
        s.listen(0)                 # Now wait for client connection.
        return s,host,port

    def clear_buffer(self):
        self.c.settimeout(0.1)
        try:
            self.c.recv(5000)
        except:
            pass
        self.c.settimeout(5)




    def start_socket(self):
        #hei = '--height {}'.format(self.map_size[0])
        #wid = '--width {}'.format(self.map_size[1])
        #cmd = 'halite.exe "--no-replay" "--no-timeout" -n 2 -vvv --width {} --height {}  "python Agent1.py" "python IdleBot.py"'.format(self.map_size[0], self.map_size[1])
        cmd = 'halite.exe "--no-replay" "--no-timeout" -n 2 -vvv --width {} --height {}  "python Agent1Socket.py" "python IdleBot.py"'.format(self.map_size[0], self.map_size[1])
        cmd = 'halite.exe "--no-replay" -n 2 -vvv --width {} --height {}  "python Agent1Socket.py" "python IdleBot.py"'.format(self.map_size[0], self.map_size[1])
        #cmd = ["halite.exe", '--players 2', '"--no-timeout"', wid, hei, '"python Agent1.py"', '"python IdleBot.py"']
        #cmd = ["halite.exe", '"--no-replay"', '-vvv', '-n 2', '"--no-timeout"', wid, hei, '"python Agent1.py"', '"python IdleBot.py"']
        #cmd = ["halite.exe", '"--no-replay"', '-vvv', '--width 8', '--height 8', '"python Agent1.py"', '"python IdleBot.py"']
        print(cmd)
        #cmd = ["python", "teste_socket.py"]
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.s, self.host, self.port = self.setupSocket()
        import time
        a = time.time()
        self.c, self.addr = self.s.accept()
        print(time.time() - a)


    def get_socket_message(self):
        try:
            msg = self.c.recv(1024)
        except:
            return None
        if(msg != b''):
            '''try:
                clientsocket.send(msg)
            except socket.timeout:
                break'''
            msg = np.frombuffer(msg,dtype = np.uint16)
            msg = msg.reshape(6,6,4)
            print(self.addr, ' >> ', msg)
        else:
            return None

        return msg


    def step(self, action):  

        msg = np.array(action, dtype = np.uint16).tobytes()
        try:
            self.c.send(msg)
        except socket.timeout:
            print("socket timeout")
            return
        #received = self.c.recv(584)
        reward = self.c.recv(4)
        done = self.c.recv(4)
        received = self.c.recv(144*4)
        reward = np.frombuffer(reward, dtype = np.uint16)[0]
        done = np.frombuffer(done, dtype = np.uint16)[0]
        received = np.frombuffer(received, dtype = np.uint16)
        print(reward)
        print(done)
        print(received)
        #reward = received[0]
        #done = received[1]
        #observation = received[2:]
        observation = received
        observation = observation.reshape(self.map_size[0],self.map_size[1], 4)



        return observation, reward, done, {}


        '''
        if(self.simulador.soloplay):
            action = self.simulador.step(action)
            observation = self.simulador.make_image()
            reward = 0
            done = False
            if(self.simulador.is_ship_in_shipyard() and action is not 4):
                reward = self.simulador.terminal_value() - self.player_halite_prev
                self.player_halite_prev = self.simulador.terminal_value()
                done = True
            if(self.simulador.terminal()):
                done = True
                #reward = self.simulador.terminal_value()

                #self.reward = reward
                #self.reset()

            return observation, reward, done, {'action': action} # observation, reward, done, info
        elif self.simulador.duoplay:
            self.simulador.step(action)
            observation, ship_layer1, ship_layer2 = self.simulador.make_image()
            reward = 0
            done = False
            if(self.simulador.terminal()):
                done = True
            else:
                for i in range(2):
                    if(self.simulador.is_ship_in_shipyard(i) and action[i] is not 4):
                        reward = self.simulador.terminal_value() - self.player_halite_prev
                        self.player_halite_prev = self.simulador.terminal_value()
                        #done = True
                    #reward = self.simulador.terminal_value()

                    #self.reward = reward
                    #self.reset()
            return (observation, ship_layer1, ship_layer2), reward, done, {} # observation, reward, done, info
        '''




    def reset(self):
        '''
        self.simulador = HaliteSimulator.HaliteGame()
        #self.simulador = HaliteSimulator.HaliteGame(prevstate = self.prevstate)
        #self.prevstate = deepcopy(self.simulador.state)
        self.player_halite_prev = 0
        if(self.simulador.duoplay):
            observation, ship_layer1, ship_layer2 = self.simulador.make_image()
            #print(self.reward)
            return observation, ship_layer1, ship_layer2
        else:
            observation = self.simulador.make_image()
            #print(self.reward)
            return observation
        '''
        self.clear_buffer()
        self.s.close()
        self.start_socket()
        reward = self.c.recv(4)
        done = self.c.recv(4)
        received = self.c.recv(144*4)
        #received = self.c.recv(1024)
        print(received)
        reward = np.frombuffer(reward, dtype = np.uint16)
        done = np.frombuffer(done, dtype = np.uint16)
        received = np.frombuffer(received, dtype = np.uint16)
        print(received)
        #reward = received[0]
        #done = received[1]
        #observation = received[2:]
        observation = received
        observation = observation.reshape(self.map_size[0],self.map_size[1], 4)
        return observation

        
        
        
    def render(self, mode='human', close=False):
        if(self.simulador.duoplay):
            observation, ship_layer1, ship_layer2 = self.simulador.make_image()

            #plt.plot(np.stack((observation[:,:,0], np.zeros(map_size), np.zeros(map_size)), axis=2))
            #teste = np.vstack((np.stack((np.zeros(HaliteSimulator.map_size), np.zeros(HaliteSimulator.map_size), observation[:,:,0].T), axis=2),
            #    np.stack((np.zeros(HaliteSimulator.map_size), np.zeros(HaliteSimulator.map_size), observation[:,:,1].T), axis=2)))
            teste = np.vstack((np.stack((np.zeros(HaliteSimulator.map_size), np.zeros(HaliteSimulator.map_size), observation[:,:,0].T / 1000), axis=2),
                np.stack((observation[:,:,2].T, np.zeros(HaliteSimulator.map_size), ship_layer1[:,:,0].T + ship_layer2[:,:,0].T), axis=2)))
            #teste = np.vstack((observation[:,:,0].T, np.stack((np.zeros((6,6)), np.zeros((6,6))), axis=2),np.stack((observation[:,:,1].T, np.zeros((6,6)), np.zeros((6,6))), axis=2)))
            teste = cv2.resize(teste,(500,1000), interpolation = cv2.INTER_NEAREST)
            cv2.imshow("title",teste)
            #plt.title("Player Halite: {} \nShip Halite: {}".format(self.simulador.state.Players[0].halite_amount, self.simulador.state.Players[0]._ships[0].halite_amount))
            #plt.show()
            cv2.waitKey(100)
            #cv2.destroyAllWindows()
            return

        else:
            observation = self.simulador.make_image()

            #plt.plot(np.stack((observation[:,:,0], np.zeros(map_size), np.zeros(map_size)), axis=2))
            #teste = np.vstack((np.stack((np.zeros(HaliteSimulator.map_size), np.zeros(HaliteSimulator.map_size), observation[:,:,0].T), axis=2),
            #    np.stack((np.zeros(HaliteSimulator.map_size), np.zeros(HaliteSimulator.map_size), observation[:,:,1].T), axis=2)))
            teste = np.vstack((np.stack((np.zeros(HaliteSimulator.map_size), np.zeros(HaliteSimulator.map_size), observation[:,:,0].T / 1000), axis=2),
                np.stack((observation[:,:,2].T, np.zeros(HaliteSimulator.map_size), observation[:,:,1].T), axis=2)))
            #teste = np.vstack((observation[:,:,0].T, np.stack((np.zeros((6,6)), np.zeros((6,6))), axis=2),np.stack((observation[:,:,1].T, np.zeros((6,6)), np.zeros((6,6))), axis=2)))
            teste = cv2.resize(teste,(500,1000), interpolation = cv2.INTER_NEAREST)
            cv2.imshow("title",teste)
            #plt.title("Player Halite: {} \nShip Halite: {}".format(self.simulador.state.Players[0].halite_amount, self.simulador.state.Players[0]._ships[0].halite_amount))
            #plt.show()
            cv2.waitKey(100)
            #cv2.destroyAllWindows()
            return

    def close(self):
        cv2.destroyAllWindows()
        #pass


