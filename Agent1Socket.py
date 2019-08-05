#!/usr/bin/env python3

import time
a = time.time()
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging
#import json
import numpy as np
import csv
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import pickle
#import logz
#import tensorflow as tf
#import tensorflow.contrib.layers as layers
#from contextlib import contextmanager
import socket

def end_game():
    asd(4125346,1432,6435) # just cause an error in the interpreter to end the game

def start_socket():
    #HOST = socket.gethostname()     # Endereco IP do Servidor
    HOST = '127.0.0.1'
    PORT = 7080            # Porta que o Servidor esta
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dest = (HOST, PORT)
    tcp.settimeout(5)
    tcp.connect(dest)
    return tcp, dest

def send_state(tcp, state, reward=0, done=0):
    logging.info("{} tamanho da mensagem state".format(len(state)))
    logging.info(state)
    logging.info("Enviando estado")
    state = state.tobytes()
    #logging.info("{} tamanho da mensagem state".format(len(state)))
    #logging.info(state)
    reward = np.array(reward, dtype=np.uint16).tobytes()
    done = np.array(done, dtype=np.uint16).tobytes()
    tcp.send(reward)
    tcp.send(done)
    tcp.send(state)
    logging.info("Estado enviado")



def receive_action(tcp):
    logging.info("Esperando acao")
    action = tcp.recv(4)
    #logging.info(action)
    action = np.frombuffer(action,dtype = np.uint16)
    logging.info("Acao recebida {}".format(action))
    #logging.info(action)
    return action
'''
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
'''

def make_image(player_id, ship_id):
    halites = np.zeros(map_size, dtype=np.uint16)
    ship_layer = np.zeros(map_size, dtype=np.uint16)
    ship_hamount = np.zeros(map_size, dtype=np.uint16)
    if(ship_id in game.players[player_id]._ships.keys()):
        ship_hamount = np.ones(map_size, dtype=np.uint16) * game.players[player_id]._ships[ship_id].halite_amount
        ship_layer[game.players[player_id]._ships[ship_id].position.x, game.players[0]._ships[ship_id].position.y] = 1000

    player_hamount = np.ones(map_size, dtype=np.uint16) * game.players[player_id].halite_amount
    shipyard_layer = np.zeros(map_size, dtype=np.uint16)


    shipyard_layer[game.players[player_id].shipyard.position.x, game.players[player_id].shipyard.position.y] = 1000
    #for x in range(self.state.game_map.width):
    #    for y in range(self.state.game_map.height):
    #        halites[x,y] = self.state.game_map[Position(x,y)].halite_amount / 100
    #print(self.state.game_map.matrix)
    halites = game.game_map.matrix.astype(np.uint16)
    #print(halites)
    state = np.stack((halites, ship_layer, shipyard_layer, ship_hamount), axis=2)
    return state




""" <<<Game Begin>>> """
# This game object contains the initial game state.
game = hlt.Game()
logging.info('{}'.format(time.time()-a))
try:
    tcp, dest = start_socket()
except:
    end_game()

logging.info('{}'.format(time.time()-a))
map_size = (game.game_map.width, game.game_map.height)
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("Agent1")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

# logging transitions
agent1 = {"state": [], "action": [], "reward": [], "next_state": []}


""" <<<Game Loop>>> """

halite_anterior = 4000

reward = 0
done = 0

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    logging.info("Recebendo frame")
    game.update_frame()
    logging.info("Frame recebido")
    game.game_map.update_matrix()
    logging.info("Matriz atualizada")
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    #logging.info(make_image(0,0))


    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    state =  make_image(0,0)
    logging.info("Imagem criada")
    if(game.turn_number > 1):
        try:
            send_state(tcp, state, reward, done)
            #logging.info("RECEBENDO ACAO")
            action = receive_action(tcp)
            #logging.info(action)
        except:
            end_game()
        try:
            action = action[0]
        except:
            print("!!!Acao que foi recebida: {}".format(action))

    for ship in me.get_ships():
        #action = agent.sess.run(agent.sy_sampled_ac, feed_dict = {agent.sy_ob_no: state[None,:]})
        logging.info(action)
        if(action == 0):
            action2 = Direction.North
        elif(action == 1):
            action2 = Direction.East
        elif(action == 2):
            action2 = Direction.South
        elif(action == 3):
            action2 = Direction.West
        elif(action == 4):
            action2 = Direction.Still
        else:
            logging.info(action)
            print("erro na acao turno {}".format(game.turn_number))

        logging.info(action2)
        command_queue.append(ship.move(action2))

    '''
    for ship in me.get_ships():
        # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
        #   Else, collect halite.
        if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
            command_queue.append(
                ship.move(
                    random.choice([ Direction.North, Direction.South, Direction.East, Direction.West ])))
        else:
            command_queue.append(ship.stay_still())
    '''




    # If the game is in the first 200 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    #if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
    #    command_queue.append(me.shipyard.spawn())

    #action = 0







    if(game.turn_number == 1):
        command_queue.append(me.shipyard.spawn())


    if(game.turn_number > 1):
        agent1["state"].append(make_image(0,0))
        agent1["reward"].append(me.halite_amount - halite_anterior)
        agent1["action"].append(action)
    if(game.turn_number>2):
        agent1["next_state"].append(make_image(0,0))

    #logging.info(agent1)
    #if(game.turn_number == constants.MAX_TURNS):
    #json.dump(agent1, open("agent1.json", 'w'), cls=NumpyEncoder)
    '''
    if(game.turn_number > 3):
        with open("agent1.csv", 'a') as f:
            f.write(str(agent1["state"][-2].tolist()) + "\t" + str(agent1["action"][-2]) + "\t" + str(agent1["reward"][-2]) + "\t" +  str(agent1["next_state"][-1].tolist()) + "\n")
    '''

    if(game.turn_number > 1):
        reward = me.halite_amount - halite_anterior
        halite_anterior = me.halite_amount




    done = 0 ##### verificar done por meio da quantidade de halite e pelo turno

    for player in game.players.values():
        if(player.halite_amount < 1000 and len(player.get_ships()) == 0):
            done = 1

    if(game.turn_number == constants.MAX_TURNS):
        done = 1

    if(done):
        send_state(tcp, state, reward, done) # sim, next_state vai acabar sendo igual ao prev state, mas nao posso resolver isso


    



    # Send your moves back to the game environment, ending this turn.
    logging.info("Enviando comandos ao halite")
    logging.info(command_queue)
    game.end_turn(command_queue)
    logging.info("Comandos enviados")

tcp.close()