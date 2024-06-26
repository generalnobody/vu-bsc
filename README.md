# BSc Project - Sparse Matrix Formats
_Note: this project is built using Python 3.11.5. It has not been tested to fully work on older versions. On Python 3.6 everything except for the PyTorch functionality works._

## Run Benchmark
To benchmark the Sparse Matrix operations, run the [main.py](./main.py) script.

### Usage
```shell
$ python main.py [-h] [--format_help] [--mode_help] -b BENCHMARK --format {coo,csr,csc,dia,bsr,lil,dok,all} --mode {add,sub,sm,mvm,mmm,tps,full} --path_a PATH_A [--path_b PATH_B] [--scalar SCALAR] [--index INDEX] [-o OUT] [-pt]
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
* **-pt, --pytorch**: use PyTorch instead of SciPy (only works with coo, csr, csc and bsr formats)

**Additional help menus:**
* **--format_help**: show additional information about the possible formats
* **--mode_help**: show additional information about the possible modes

### Example

Using SciPy:
```shell
$ python main.py --format all --mode full --path_a sample.mtx --path_b sample2.mtx --scalar 10 --index 1 -o output.json
```

Or using PyTorch:
```shell
$ python main.py --format all --mode full --path_a sample.mtx --path_b sample2.mtx --scalar 10 --index 1 -o output_pt.json -pt
```

## Get Plotted Results
To plot the results and get additional statistics, run the [results.py](./results.py) script.

### Usage
```shell
$ python results.py [-h] -f FILE [-ptf PYTORCH_FILE] [-o OUTPUT] [-fmt]
```

**Options:**
* **-h, --help**: shows the help message
* **-f, --file**: path to JSON file generated using [main.py](./main.py)
* **-ptf, --pytorch_file**: path to JSON file generated with pytorch benchmarking
* **-o, --output**: specify the folder to which to save the generated plot(s) (default: ./plots)
* **-fmt, --format**: specify the output format for the generated plot(s) (default: pdf)

### Example
```shell
$ python results.py -f output.json --plot both -o ./plots
```

## Find Memory Usage
To compare the theoretical memory usage to the actual memory usage of a sparse matrix loaded into memory, run the [memory.py](./memory.py) script.

### Usage
```shell
$ python memory.py [-h] -i INPUT [-o OUTPUT]
```

**Options:**
* **-h, --help**: shows the help message
* **-i, --input**: path to input file (mtx format) (required)
* **-o, --output**: CSV file to output result to; if not specified, only prints result to stdout (optional)

### Example

```shell
$ python memory.py sample.mtx
```

## Matrix Selection
In this project, in the [./matrices](./matrices) folder, there are sample matrices from [SuiteSparse](https://sparse.tamu.edu/) that were used in getting the results for the final thesis paper. The aim was to find matrices that would allow to test the different Sparse Matrix formats as extensively as possible, so I chose a matrix that had diagonals, a matrix that had blocks, as well as matrices that had a more "random" distribution of points.

These matrices can be plotted into a plot with subplots using the [sparse_plot.py](./sparse_plot.py) script.

### Usage

```shell
$ python sparse_plot.py [-h] -f FILE [FILE ...] -o OUTPUT
```

**Options:**
* **-h, --help**: shows the help message
* **-f, --file**: path to MatrixMarket file(s) (multiple possible)
* **-o, --output**: file to output the plot to (any format possible, including PDF, EPS, JPG, PNG)

### Example
```shell
$ python sparse_plot.py -f matrices/*.mtx -o matrices/matrices.eps
```

### Diagonals
The [Trefethen_700.mtx](./matrices/Trefethen_700.mtx) matrix has several diagonals, so it should theoretically benefit from being loaded into the DIA format.

### Blocks
The [Erdos02.mtx](./matrices/Erdos02.mtx) has extremely dense data along the left and top edges of the matrix, while the rest is empty. This should theoretically be dividable into blocks, which should benefit from being loaded into the BSR format.

### Random
The other two matrices, [ash219.mtx](./matrices/ash219.mtx) and [mk12-b2.mtx](./matrices/mk12-b2.mtx), have a more or less random distribution. These should therefore show worse performance in the DIA and BSR formats, but comparatively better performance in the other formats. One is larger, being able to showcase differences in performance in case of larger matrices, while the other is smaller, being able to showcase the opposite.