from ete3 import Tree
from pastml.tree import annotate_dates, DATE


def cut_tree(tree, threshold_date):
    forest = []
    todo = [tree]
    while todo:
        n = todo.pop()
        date = getattr(n, DATE)
        if date >= threshold_date:
            parent = n.up
            n.detach()
            n.dist = date - threshold_date
            forest.append(n)
            if parent and len(parent.children) == 1:
                child = parent.children[0]
                if not parent.is_root():
                    grandparent = parent.up
                    grandparent.remove_child(parent)
                    grandparent.add_child(child, dist=child.dist + parent.dist)
                else:
                    child.dist += parent.dist
                    child.detach()
                    tree = child
        else:
            todo.extend(n.children)
    print("Cut the tree into a root tree of {} tips and {} {}-on trees of {} tips in total"
          .format(len(tree), len(forest), threshold_date, sum(len(_) for _ in forest)))
    return tree, forest


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--tree', type=str, default='/home/azhukova/projects/bdpn/flu/NA.dated.nwk')
    parser.add_argument('--min_date', type=float, default=2023.6657534246576)
    parser.add_argument('--max_date', type=float, default=2024.4153005464482)
    parser.add_argument('--forest', type=str, default='/home/azhukova/projects/bdpn/flu/NA.dated.202324.nwk')
    params = parser.parse_args()

    tree = Tree(params.tree)
    annotate_dates([tree])

    tree, _ = cut_tree(tree, params.max_date)
    _, forest = cut_tree(tree, params.min_date)

    with open(params.forest, 'w+') as f:
        for root in forest:
            f.write('{}\n'.format(root.write(format_root_node=True, format=5, features=[DATE])))

    for root in forest:
        print([_.name for _ in root])
