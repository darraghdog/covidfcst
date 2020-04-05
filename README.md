## John Hopkins COVID county forecasting

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

```

### To predict every county for next two days

Run the below command :  
```
python train.py --data data/covid-19-data/ --rootpath . --preddays 2 --test F
```
The output can be seen with date steamp