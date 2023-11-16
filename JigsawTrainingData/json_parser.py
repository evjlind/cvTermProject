import json
import numpy

pieces_dir = "./pieces/test1/"

file = open(pieces_dir + 'index.json')
data = json.load(file)
file.close()
coords = []
for i in data['piece_properties']:
    coord = (i['x'],i['y'],i['id'])
    print(coord)
    coords.append(coord)

coords = sorted(coords,key=lambda x:x[1])
print(coords)