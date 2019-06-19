from pyspark import SparkContext
import sys
import time
import csv

s=time.time()

def hash_function(each_hash,a):
    return min([((each_hash[0]*x+each_hash[1])%each_hash[2])%m for x in a[1]])

def j_simi(a):   # Calculate Jaccard Similarity
    a1 = set(characteristics_matrix[bus_num[a[0]]][1])
    a2 = set(characteristics_matrix[bus_num[a[1]]][1])
    intersection = len(a1&a2)
    union = len(a1|a2)
    return (a[0],a[1],intersection/union)

def sig(x):   # Generate signatures
    res = []
    for i in range(b):
        res.append(((i, tuple(x[1][i * r:(i + 1) * r])), [x[0]]))
    return res

def cands(a):   # Generate candidates
    r=[]
    bus=list(a[1])
    bus.sort()
    for i in range(len(a[1])):
        for j in range(i+1,len(a[1])):
            r.append(((bus[i],bus[j]),1))
    return r

sc=SparkContext('local[*]','jaccard')

# Read training file
data_raw=sc.textFile(sys.argv[1], minPartitions=2)
# Remove the header from training data
first_row=data_raw.first()
data=data_raw.filter(lambda a: a != first_row).map(lambda a: a.split(","))

# Get list of businesses for training data
b=data.map(lambda a: (a[1],1)).reduceByKey(lambda a,b: a).map(lambda a: a[0])
business=list(b.collect())
business.sort()

# Get list of users for training data
usr = data.map(lambda a: (a[0],1)).reduceByKey(lambda a,b: a).map(lambda a: a[0])
users=list(usr.collect())
users.sort()

# m,n,b,r are parameters for the algotithm
m=len(users)
hash_values = [[11, 43, 107], [2, 63, 97], [17, 23, 769], [181, 251, 421],[13, 37, 3079], [14, 91, 1543], [73, 803, 49157], [1, 101, 193], [913, 901, 24593], [91, 29, 12289], [8, 119, 389], [41, 443, 6311], [3, 79, 53], [387, 552, 98317], [71, 67, 6151], [887, 2281, 7817]]
n=len(hash_values)
b=8
r=int(n/b)
bus_num={}
user_num={}
for num, bus_name in enumerate(business):
    bus_num[bus_name]=num
for num, user_name in enumerate(users):
    user_num[user_name]=num
user_dict=sc.broadcast(user_num)

# Create characteristics matrix
charmat=data.map(lambda a: (a[1], [user_dict.value[a[0]]])).reduceByKey(lambda a,b: a+b)
char_matrix=charmat.sortBy(lambda a: a[0])
characteristics_matrix=char_matrix.collect()

# Generate signatures
signatures=char_matrix.map(lambda a: (a[0], [hash_function(each_hash,a) for each_hash in hash_values]))

# Generate candidates
candidate1=signatures.flatMap(sig).reduceByKey(lambda a,b: a+b).filter(lambda a: len(a[1])>1)
candidate=candidate1.flatMap(cands).reduceByKey(lambda a,b: a).map(lambda a: a[0])
result = candidate.map(j_simi).filter(lambda a: a[2]>=0.5).sortBy(lambda a: a[1]).sortBy(lambda a: a[0])
result1 = result.collect()

# Write similarity to output csv file
with open(sys.argv[2], mode='w') as ofile:
    ofile_w = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    ofile_w.writerow(['business_id_1', ' business_id_2', ' similarity'])
    if(result):
        for pair in result1:
            ofile_w.writerow([pair[0], pair[1], pair[2]])
ofile.close()
print("Time:",str(time.time()-s))

