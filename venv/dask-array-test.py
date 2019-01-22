import dask.array as da

#using arange to create an array with values from 0 to 10
X = da.arange(11, chunks=5)
X.compute()
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10])

#to see size of each chunk
X.chunks
((5, 5, 1),)