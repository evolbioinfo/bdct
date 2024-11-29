from ete3 import Tree

tree = Tree('/home/azhukova/projects/bdpn/flu/NA.nwk')
for n in tree.traverse():
    if not n.is_root():
        n.dist = float(n.date) - float(n.up.date)
    else:
        n.dist = 0

tree.write(outfile='/home/azhukova/projects/bdpn/flu/NA.dated.nwk', features=['date'])
