""" Main script """
import argparse

EXACT = 'e'
APPROX = 'a'

parser = argparse.ArgumentParser()
parser.add_argument("fname", help="Filename to read")
parser.add_argument("-m", "--mode", choices=[EXACT, APPROX])
# group = parser.add_mutually_exclusive_group()
# group.add_argument("-e", "--exact", action="store_true",
#                    help="Exact computation")
# group.add_argument("-a", "--approx", action="store true",
#                    help="Approximate computation")

args = parser.parse_args()

if args.mode is EXACT:
    # ToDo: get exact distribution
    pass
else:
    # ToDo: get approx distribution
    # anf0 by default, other methods passed as parameters?
    pass

# ToDo:  Get statistics from the distribution
# ToDo: Print statistics
