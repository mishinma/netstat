""" Main script """
import argparse


def run_exact(graph, directed=True):
    pass

parser = argparse.ArgumentParser()
parser.add_argument("fname", help="Filename to read")
parser.add_argument("-m", "--mode", choices=["e",  "a"])
# group = parser.add_mutually_exclusive_group()
# group.add_argument("-e", "--exact", action="store_true",
#                    help="Exact computation")
# group.add_argument("-a", "--approx", action="store true",
#                    help="Approximate computation")
parser.add_argument('--example', nargs='?', const=1, type=int, default=1)

args = parser.parse_args()
if args.verbosity:
    print "verbosity turned on"