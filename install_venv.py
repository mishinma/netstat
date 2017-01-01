import tarfile

tar = tarfile.open("sw/virtualenv-15.1.0.tar.gz")
tar.extractall()
tar.close()

import sys
import os

path = os.path.abspath("sw")
assert os.path.exists(path)

win32 = sys.platform.startswith('win')
mac = sys.platform == 'darwin'

if win32:
    raise OSError("Windows is not supported")
else:
    if mac:
        print "Mac platform detected. In case of compilation issues, try exporting numpy's path to "\
        "CFLAGS and run again. Example:\n"\
        "export CFLAGS=\"-I /usr/local/lib/python2.7/site-packages/numpy/core/include $CFLAGS\""
    os.system("%s virtualenv-15.1.0/virtualenv.py --system-site-packages .venv" % sys.executable)
    os.system(".venv/bin/pip install -r requirements.txt")
    # Packages also available in sw
    # os.system(".venv/bin/pip install  -r requirements.txt --no-index "
    #           "--upgrade -f file://{}".format(path))
    os.system(".venv/bin/python setup.py build_ext --inplace")
    os.system(".venv/bin/python setup.py install")
