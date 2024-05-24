# BSc Project - Sparse Matrix Formats
_Note: this project is built using Python 3.11.5. It has been tested to work on Python 3.6, but not on older versions. Shell instructions in this file use the `python` command, but if Python 2.7 is also installed, may require the `python3` command, depending on settings._

## Run Benchmark
To benchmark the Sparse Matrix operations, run the [main.py](./main.py) script.

### Usage
```shell
$ python main.py [-h] [--format_help] [--mode_help] -b BENCHMARK --format {coo,csr,csc,dia,bsr,lil,dok,all} --mode {add,sub,sm,mvm,mmm,tps,full} --path_a PATH_A [--path_b PATH_B] [--scalar SCALAR] [--index INDEX] [-o OUT]
```

**Main options:**
* **-h, --help**: show the help message
* **-b, --benchmark**: set the number of times to benchmark the chosen mode(s) (minimum 1)
* **--format**: choose sparse matrix format(s) to use (required)
* **--mode**: choose the function(s) to benchmark (required)
* **--path_a**: path to the main matrix to be used for the benchmark (mtx format) (required)
* **--path_b**: path to the secondary matrix to be used for the benchmark (mtx format) (required for mores add, sub and mmm)
* **--scalar**: scalar function used for the benchmark (required for mode sm)
* **--index**: index of the row in the matrix to select as vector (optional for mode mvm; if not chosen, selected randomly)
* **-o, --out**: file to save the result to (JSON format)

**Additional help menus:**
* **--format_help**: show additional information about the possible formats
* **--mode_help**: show additional information about the possible modes

### Example
```shell
$ python main.py --format all --mode full --path_a sample.mtx --path_b sample2.mtx --scalar 10 --index 1 -o output.json
```

## Get Plotted Results
To plot the results and get additional statistics, run the [results.py](./results.py) script.

### Usage
```shell
$ python results.py [-h] -f FILE --plot {both,format,mode} [-o OUTPUT] [-s]
```

**Options:**
* **-h**: shows the help message
* **-f**: path to JSON file generated using [main.py](./main.py)
* **--plot**: select whether to plot based on function or format, or both (both recommended)
* **-o**: specify the folder to which to save the generated plot(s) (default: ./plots)
* **-s**: show the plots as they are generated (not recommended when provided JSON file contains multiple results)

### Example
```shell
$ python results.py -f output.json --plot both -o ./plots
```

## Find Memory Usage
To compare the theoretical memory usage to the actual memory usage of a sparse matrix loaded into memory, run the [memory.py](./memory.py) script.

### Usage
```shell
$ python memory.py [-h] -i INPUT
```

**Options:**
* **-h**: shows the help message
* **-i**: path to input file (mtx format) (required)

### Example

```shell
$ python memory.py sample.mtx
```

## Profiling
Provides more detailed profiling results for any operation possible in [main.py](./main.py). To get these results, run the [profiling.py](./profiling.py) script.

### Usage
```shell
$ python profiling.py [-h] [--format_help] [--mode_help] --format {coo,csr,csc,dia,bsr,lil,dok} --mode {add,sub,sm,mvm,mmm,tps} --path_a PATH_A [--path_b PATH_B] [--scalar SCALAR] [--index INDEX]
```

Same options as shown in [main.py's usage](#usage), excluding the **-b** and **-o** arguments.

### Example
```shell
$ python profiling.py --format coo --mode add --path_a sample.mtx --path_b sample2.mtx
```