Baselines
---------

We provide 2 files to reproduce the baseline results. We tested both the scripts on the Leonhard cluster. We assume the following file structure
```
../../data/data_train.csv
../../data/sampleSubmission.csv
./ga_svd.py
./svdpp.py
```

For reproducing the results of General Avg, and SVD run
```
module load python_gpu/3.7.1
bsub -W 3:00 -R "rusage[ngpus_excl_p=1,mem=16392]" -o output.out python ga_svd.py
```

For reproducing the results of SVDPP run
```
module load python_gpu/3.7.1
bsub -W 80:00 -R "rusage[ngpus_excl_p=1,mem=16392]" -o output.out python svdpp.py
```
For reproducing the results of Bayesian SVDPP

set ```bayes_by_backprop = True # (line 1 svdpp.py) ```

```
module load python_gpu/3.7.1
bsub -W 80:00 -R "rusage[ngpus_excl_p=1,mem=16392]" -o output.out python svdpp.py
```

The above scripts would also evaluate the model on sampleSubmission.csv and generate corresponding CSV files.
