# Netstat

Netstat is a tool for computing statistics (mean, median, 
diameter and effective diameter) for large networks.

## Installation

1. Change directory to the package directory and run:

    ```
    python install_venv.py
    ```
    
    The script will create a virtual environment `.venv`
    inside the package and install the required packages using pip.
    It will also build and install `netstat` itself.
    
2. Activate the virtual environment:

    ```
    . ./venv/bin/activate
    ```
    
    Alternatively, you can create your own virtual environment or
    even use sistem packages (not recommended). All required packages
    are also available in `netstat/sw`. You can install them from there:
    
    ```
    pip install -r requirements.txt --no-index --upgrade -f file://<path_to_sw>"
    ```
    
## Usage

   To run Netstat in the command line you should use the script `netstat-run`. 
   The usage is as follows:
   ```
   netstat-run [-h] [-c] [-s] [-u] fname {exact,randp,rands,anf} ...
   ```
   
   Positional arguments:
       * `fname` - filename to read. The file **must** be "cleaned" before
       running the computations. To clean the file provide the `-c` option
       * `{exact,randp,rands,anf}` - choose the algorithm you want to use;
       in respective order: exact computation, sample random pairs, 
       sample random sources, ANF (ANF-0 from [1]). 
   
   Optional arguments:
       * `-s` - use the single-core version (slower)
       * `-c` - clean the file
       * `-u` - treat the graph as undirected
       * `-h` - help
   
   Algoritm-specific optional arguments:
   
   * `exact` usage:
   ```
   netstat fname exact [-h]
   ```
   * `randp` usage:
   ```
   netstat fname randp [-h] [-k K | -p P]
   ```  
   ..* `-k` - sample size (number of pairs)
   ..* `-p` - sample size (precentage of the total number nodes)
   By default, the sample size is set to 10% of the total number of nodes.  
   * `rands` usage:
   ```
   netstat fname rands [-h] [-k K | -p P]
   ``` 
   Optional arguments are the same as `randp`.
   * `anf` usage:
   ```
   netstat fname anf [-h] [-r R] [-k K] [-h H]
   ```
   ..* `-r` - number of bits to add to bitstring of length `log(n)`,
   where `n` is number of nodes. (Default: 0)
   ..* `-k` - number of parallel approximations. (Default: 3)
   ..* `-h` - number of distances to approximate. (Default: 20)
   
### Example usage:
```
(.venv) misha@misha:~/dev/netstat$ netstat-run ~/docs/netstat/data/wiki-Vote-clean.txt -u anf -k 10
```

To get help message run:
```
netstat-run test.in
# or
netstat-run test.in anf -h
```

**Note**: cleaning the file by default will create another file
with suffix "-clean" in the same directory as the original file.
Alternatively, you can use the script `clean`:
```
clean fname_old [fname_new]
```
   
    
## Platform Issues

1. The script `install_venv.py` works only on Linux and OSX. 
   The package itself should work on Windows, but that was not
   tested and, therefore, is not recommended.

2. There is a [known issue](https://github.com/hmmlearn/hmmlearn/issues/43)  with OSX and numpy headers.
   When building the package you can get 
   `fatal error: 'numpy/arrayobject.h' file not found`.
   We attempted to resolve this issue in `install_venv.py`. 
   
   Please contact the authors if you have problems with installing or
   using the package.

## Authors

Mikhail Mishin & Max Reuter
   
## References
    
   [1] C. R. Palmer, P. B. Gibbons, and C. Faloutsos. ANF: A fast and scalable tool for data mining in
    massive graphs. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge
    discovery and data mining, pages 81–90. ACM, 2002.




 