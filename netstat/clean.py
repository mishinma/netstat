import os
import sys
import subprocess

# ToDo: Write it as a bash script?

class FileFormatError(Exception):
    pass


class Cleaner(object):

    def __init__(self):
        self.vtx_map = dict()
        self.vtx_cnt = 0

    def __call__(self, vtx_id):
        try:
            vtx_id_new = self.vtx_map[vtx_id]
        except KeyError:
            self.vtx_map[vtx_id] = self.vtx_cnt
            vtx_id_new = self.vtx_cnt
            self.vtx_cnt += 1
        return vtx_id_new


def clean_file(fname_old, fname_new):

    with open(fname_old, 'r') as f_old:
        with open(fname_new, 'w') as f_new:

            clnr = Cleaner()
            edge_cnt = 0

            for line in f_old:

                # Skip comments
                if line.startswith('#'):
                    continue

                id_from, id_to = map(int, line.split())
                id_from_new = clnr(id_from)
                id_to_new = clnr(id_to)
                f_new.write("{} {}\n".format(id_from_new, id_to_new))
                edge_cnt += 1

    num_nodes = clnr.vtx_cnt
    num_edges = edge_cnt

    subprocess.call(
        "sort -n -k1,1 -k2,2 -o {fname_new} {fname_new}".format(fname_new=fname_new), shell=True
    )

    subprocess.call(
        "echo '{num_nodes} {num_edges}' | cat - {fname_new} > temp && mv temp {fname_new}" \
            .format(num_nodes=num_nodes, num_edges=num_edges, fname_new=fname_new),
        shell=True
    )


if __name__ == '__main__':

    fname_old = sys.argv[1]
    root, ext = os.path.splitext(fname_old)
    fname_new = root + '-clean' + ext

    clean_file(fname_old, fname_new)


