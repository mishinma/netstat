import os
import sys
import subprocess


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

            for line in f_old:

                # Skip comments
                if line.startswith('#'):
                    continue

                id_from, id_to = line.split()
                id_from_new = clnr(id_from)
                id_to_new = clnr(id_to)
                f_new.write("{} {}\n".format(id_from_new, id_to_new))




    subprocess.call(
        "sort -n -u -k1,1 -k2,2 -o {fname_new} {fname_new}".format(fname_new=fname_new), shell=True
    )

    num_nodes = clnr.vtx_cnt

    with open(fname_new, 'r') as f_new:
        for i, l in enumerate(f_new, start=1):
            pass
    num_edges = i


    subprocess.call(
        "echo '#!clean\n{num_nodes} {num_edges}' | cat - {fname_new} > temp && mv temp {fname_new}" \
            .format(num_nodes=num_nodes, num_edges=num_edges, fname_new=fname_new),
        shell=True
    )


def main():
    fname_old = sys.argv[1]
    if len(sys.argv) > 2:
        fname_new = sys.argv[2]
    else:
        root, ext = os.path.splitext(fname_old)
        fname_new = root + '-clean' + ext
    clean_file(fname_old, fname_new)


if __name__ == '__main__':
    main()


