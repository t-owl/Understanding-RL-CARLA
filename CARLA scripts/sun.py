import glob
import os
import sys
try:
    sys.path.append(glob.glob('/home/dantek/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse

argparser = argparse.ArgumentParser(
        description=__doc__)
argparser.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',
    help='IP of the host server (default: 127.0.0.1)')
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
argparser.add_argument(
    '-s', '--speed',
    metavar='FACTOR',
    default=1.0,
    type=float,
    help='rate at which the weather changes (default: 1.0)')
args = argparser.parse_args()

args = argparser.parse_args()
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()




weather = carla.WeatherParameters(
    sun_altitude_angle=50.0)

world.set_weather(weather)
