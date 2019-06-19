Yelp Recommendation System

Task 1 : Jaccard based LSH
 - implement the Locality Sensitive Hashing algorithm with Jaccard similarity using yelp_train.csv
 - focus on “0 or 1” ratings rather than the actual ratings/stars from the users
 - need to identify similar businesses whose similarity >= 0.5
 - hash functions are:
  f(x)= (ax + b) % m or f(x) = ((ax + b) % p) % m
  where p is any prime number and m is the number of bins.
 - divide the matrix into b bands with r rows each, where b x r = n (n is the number of hash functions)
 
 Task 2 : Recommendation System
  Case 1: Model-based CF recommendation system with Spark MLlib
  Case 2: User-based CF recommendation system
  Case 3: Item-based CF recommendation system
  Case 4: Item-based CF recommendation system with Jaccard based Locality Sensitive Hashing
