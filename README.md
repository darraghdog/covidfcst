## John Hopkins COVID county forecasting

### Install
```
git clone https://github.com/darraghdog/covidfcst
cd covidfcst
```

### Connect to data set

To connect to the data set first time, clone the repo.   
```
cd data
https://github.com/nytimes/covid-19-data
```

### Refresh data daily
To refresh the data daily :   

```
cd data/covid-19-data
git pull
```

### To test MAE and RMSE on last 2 days of every county

Run the below command :  
```
python train.py --data data/covid-19-data/ --rootpath . --preddays 2 --test T
```

The output should be like :  
```
Fitting 3 folds for each of 1 candidates, totalling 3 fits
[Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:   23.2s finished
Test MAE  : 9.6478
Test RMSE : 5182.0586
```

### To predict every county for next two days

Run the below command :  
```
python train.py --data data/covid-19-data/ --rootpath . --preddays 2 --test F
```
The output can be seen in `out` directory with a date stamp.