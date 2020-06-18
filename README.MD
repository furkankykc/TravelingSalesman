## TRAVELING SALESMAN PROBLEM
### Install
```.shell script
pip install traveling-salesman
```

### Usage
```python
from TSM.TSM import greedy,smallest_increase,read_all,find_closest_path
nodes = read_all('resources/sehir_xy')

smallest_book, distance_result = smallest_increase(nodes, start_index=45, end_index=51, plot=True, plot_annotate=True)
greedy_book, distance_result = greedy(nodes, start_index=45, end_index=51, plot=False)
find_closest_path(greedy_book, 12, 5, plot=True)
# find_closest_path(smallest_book, 12, 5, plot=True, show_route=True, suptitle='Smallest Increase Algorithm Route')
```
### Smallest Increase Algorihm
```python
greedy_book, distance_result = greedy(nodes, start_index=45, end_index=51, plot=False)
```
### Result
![greedyfcp](https://media.giphy.com/media/VCsDjsWFzEVyhNwb90/giphy.gif)
### Greedy Algorihm
```python
greedy_book, distance_result = greedy(nodes, 45, 51, plot=False)
```
### Result
![greedyfcp](https://media.giphy.com/media/WU70bFnCg9GcxrVvWY/giphy.gif)

### Find Closest Path
```python
greedy_book, distance_result = greedy(nodes, start_index=45,end_index=51, plot=False)
find_closest_path(greedy_book, 12, 5, plot=True)
```
### Result
![greedyfcp](https://raw.githubusercontent.com/furkankykc/TravelingSalesman/master/images/fcpgreedy.png)