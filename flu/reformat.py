import re

lines = open('/home/azhukova/projects/bdpn/flu/flu_na.nwk', 'r').read()
print(lines)

i = len(lines)

while True:
    i = lines.rfind('[&&NHX:', 0, i)
    if i == -1:
        break
    j = lines.find(']', i)
    annotation = lines[i: j + 1]
    if lines[j + 1] == ':':
        dist = next(re.finditer(r'[:]\d+[\.\d]*', lines[j:]))[0]
        lines = lines[:i] + dist + annotation + lines[j + 1 + len(dist):]

with open('/home/azhukova/projects/bdpn/flu/NA.nwk', 'w+') as f:
    f.write(lines)