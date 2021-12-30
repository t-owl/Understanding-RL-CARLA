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
import random


args = argparser.parse_args()
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
current_map = world.get_map()

print(self.map.get_spawn_points())
# # spawn_transforms will be a list of carla.Transform
# spawn_transforms = current_map.get_spawn_points()
# # get a single random spawn transformation over the map
# random_spawn = random.choice(spawn_transforms)