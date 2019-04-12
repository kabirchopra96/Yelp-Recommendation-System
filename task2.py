from pyspark import SparkContext
from math import sqrt
import csv
import sys
import time
from pyspark.mllib.recommendation import ALS, Rating

s = time.time()

def case1 (train_file,test_file,out_file):
    def abc(x):
        if (x[0] not in user_num):
            a = un_a[x[0]]
        else:
            a = user_num[x[0]]
        if (x[1] not in bus_num):
            b = bn_a[x[1]]
        else:
            b = bus_num[x[1]]

        return (a, b)

    def rates(a):
        if (a[2] < 1):
            rate = 1.0
        elif (a[2] > 5):
            rate = 5.0
        else:
            rate = a[2]
        return ((a[0], a[1]), rate)

    data_raw = sc.textFile(train_file)
    first_row = data_raw.first()
    data1 = data_raw.filter(lambda a: a != first_row).map(lambda a: a.split(","))
    test_raw = sc.textFile(test_file).map(lambda x: x.split(","))
    first_row_test = test_raw.first()
    test_data = test_raw.filter(lambda a: a != first_row_test)
    b = data1.map(lambda a: (a[1], 1)).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    business = list(b.collect())
    usr = data1.map(lambda a: (a[0], 1)).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    users = list(usr.collect())
    bus_num = {}
    user_num = {}
    bus_num1 = {}
    user_num1 = {}
    for num, bus_name in enumerate(business):
        bus_num[bus_name] = num
        bus_num1[num] = bus_name
    for num, user_name in enumerate(users):
        user_num[user_name] = num
        user_num1[num] = user_name
    un = len(user_num)
    bn = len(bus_num)
    data = data1.map(lambda a: ((user_num[a[0]], bus_num[a[1]]), float(a[2])))
    tb = test_data.map(lambda a: (a[1], 1)).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    tbusiness = list(tb.collect())
    tusr = test_data.map(lambda a: (a[0], 1)).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    tusers = list(tusr.collect())
    un_a = {}
    un_b = {}
    bn_b = {}
    bn_a = {}
    for num, bus_name in enumerate(tbusiness):
        if (bus_name not in bus_num):
            bn_a[bus_name] = num + bn + 1
            bn_b[num + bn + 1] = bus_name
    for num, user_name in enumerate(tusers):
        if (user_name not in user_num):
            un_a[user_name] = num + un + 1
            un_b[num + un + 1] = user_name
    test_map = test_data.map(abc)
    test_data = test_map.map(lambda x: (x, None))
    ratings = data.map(lambda a: Rating(a[0][0], a[0][1], a[1]))
    numIterations = 10
    rank = 4
    model = ALS.train(ratings, rank, numIterations)
    predic = model.predictAll(test_map).map(rates)
    predic_new = test_data.subtractByKey(predic).map(lambda a: (a[0], 3.0))
    predictions = sc.union([predic, predic_new])
    with open(out_file, mode='w') as ofile:
        ofile_w = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ofile_w.writerow(['user_id', ' business_id', ' prediction'])
        if (predictions):
            for e in predictions.collect():
                if (e[0][0] not in user_num1 and e[0][1] not in bus_num1):
                    ofile_w.writerow([un_b[e[0][0]], bn_b[e[0][1]], e[1]])
                elif (e[0][0] not in user_num1):
                    ofile_w.writerow([un_b[e[0][0]], bus_num1[e[0][1]], e[1]])
                elif (e[0][1] not in bus_num1):
                    ofile_w.writerow([user_num1[e[0][0]], bn_b[e[0][1]], e[1]])
                else:
                    ofile_w.writerow([user_num1[e[0][0]], bus_num1[e[0][1]], e[1]])
    ofile.close()

def case2 (train_file,test_file,out_file):
    def predict(train_data, train_dict2_a, current_user, current_business):
        train_dictionary_1=train_data.value
        train_dict2=train_dict2_a.value
        if (train_dictionary_1.get(current_user)):
            current_user_businesses=list(train_dictionary_1.get(current_user))
            current_user_ratings=train_dictionary_1.get(current_user)
            sum_ratings=sum(current_user_ratings.values())
            ratings_diff=[]
            weights=[]
            if(train_dict2.get(current_business)!=None):
                rated_by_users=list(train_dict2.get(current_business))
                if (rated_by_users):
                    average_of_current_user=sum_ratings/len(list(current_user_ratings))
                    for i in range(len(rated_by_users)):
                        current_user_total_rating=0
                        other_user_total_rating=0
                        k=0
                        c_b_p=[]
                        o_b_p=[]
                        current_rating=train_dictionary_1[rated_by_users[i]].get(current_business)
                        while (k<len(current_user_businesses)):
                            if (train_dictionary_1[rated_by_users[i]].get(current_user_businesses[k])):
                                current_user_total_rating+=train_dictionary_1[current_user].get(current_user_businesses[k])
                                other_user_total_rating+=train_dictionary_1[rated_by_users[i]].get(current_user_businesses[k])
                                o_b_p.append(train_dictionary_1[rated_by_users[i]].get(current_user_businesses[k]))
                                c_b_p.append(train_dictionary_1[current_user].get(current_user_businesses[k]))
                            k+=1
                        k=0
                        if(c_b_p):
                            den_first = 0
                            den_second = 0
                            w = 0
                            num = 0
                            den = 0
                            first = 0
                            second = 0
                            average_other_user=other_user_total_rating/len(o_b_p)
                            averagecurrent_user = current_user_total_rating / len(c_b_p)
                            for i in range(len(c_b_p)):
                                first=c_b_p[i]-averagecurrent_user
                                second=o_b_p[i]-average_other_user
                                num+=(first*second)
                                den_first+=(first*first)
                                den_second+=(second*second)
                            den=sqrt(den_first)*sqrt(den_second)
                            if (den!=0):
                                w = num/den
                            ratings_diff.append((current_rating-average_other_user)*w)
                            weights.append(w)
                    num1=sum(ratings_diff)
                    den1=sum(abs(each) for each in weights)
                    rating= -1
                    if(num1!=0 and den1!=0):
                        rating=(num1/den1)
                        rating+=average_of_current_user
                        if (rating<1.0):
                            rating=1.0
                        elif (rating>5.0):
                            rating=5.0
                        return (current_user, current_business, str(rating))
                    else:
                        rating = average_of_current_user
                        return (current_user, current_business, str(rating))
                else:
                    return (current_user, current_business, str("3.0"))
            else:
                average_of_current_user = sum_ratings / len(list(current_user_ratings))
                return (current_user, current_business, str(average_of_current_user))

        else:
            return (current_user, current_business, str("3.0"))

    data_raw=sc.textFile(train_file , minPartitions=2)
    first_row=data_raw.first()
    train_data=data_raw.filter(lambda a: a != first_row).map(lambda a: a.split(","))
    test_raw=sc.textFile(test_file).map(lambda x: x.split(","))
    first_row_test=test_raw.first()
    test_data=test_raw.filter(lambda a: a!=first_row_test)
    train_dictionary_1=train_data.map(lambda a: ((a[0]),((a[1]),float(a[2])))).groupByKey().mapValues(dict).collectAsMap()
    train_dictionary_2=train_data.map(lambda a: ((a[1]),((a[0]),float(a[2])))).groupByKey().mapValues(dict).collectAsMap()
    train_dict1=sc.broadcast(train_dictionary_1)
    train_dict2=sc.broadcast(train_dictionary_2)
    predictions=test_data.map(lambda a: predict(train_dict1, train_dict2 ,a[0], a[1]))
    with open(out_file, mode='w') as ofile:
        ofile_w = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ofile_w.writerow(['user_id', ' business_id', ' prediction'])
        if(predictions):
            for i in predictions.collect():
                ofile_w.writerow([i[0], i[1], str(i[2])])
    ofile.close()

def case3 (train_file,test_file,out_file):
    def predict(train_data, train_dict2_a, current_user, current_business):
        train_dictionary_1=train_data.value
        train_dictionary_2=train_dict2_a.value
        if (train_dictionary_2.get(current_business)):
            current_business_users=list(train_dictionary_2.get(current_business))
            current_business_ratings=train_dictionary_2.get(current_business)
            sum_ratings=sum(current_business_ratings.values())
            ratings_diff=[]
            weights=[]
            if(train_dictionary_1.get(current_user)!=None):
                rated_by_users=list(train_dictionary_1.get(current_user))
                if (rated_by_users):
                    average_of_current_business=sum_ratings/len(list(current_business_ratings))
                    for i in range(len(rated_by_users)):
                        current_business_total_rating=0
                        other_business_total_rating=0
                        k=0
                        c_b_p=[]
                        o_b_p=[]
                        current_rating=train_dictionary_2[rated_by_users[i]].get(current_user)
                        while (k<len(current_business_users)):
                            if (train_dictionary_2[rated_by_users[i]].get(current_business_users[k])):
                                current_business_total_rating+=train_dictionary_2[current_business].get(current_business_users[k])
                                other_business_total_rating+=train_dictionary_2[rated_by_users[i]].get(current_business_users[k])
                                o_b_p.append(train_dictionary_2[rated_by_users[i]].get(current_business_users[k]))
                                c_b_p.append(train_dictionary_2[current_business].get(current_business_users[k]))
                            k+=1
                        k=0
                        if(c_b_p):
                            den_first = 0
                            den_second = 0
                            w = 0
                            num = 0
                            den = 0
                            first = 0
                            second = 0
                            average_other_business=other_business_total_rating/len(o_b_p)
                            averagecurrent_business = current_business_total_rating / len(c_b_p)
                            for i in range(len(c_b_p)):
                                first=c_b_p[i]-averagecurrent_business
                                second=o_b_p[i]-average_other_business
                                num+=(first*second)
                                den_first+=(first*first)
                                den_second+=(second*second)
                            den=sqrt(den_first)*sqrt(den_second)
                            if (den!=0):
                                w = num/den
                            ratings_diff.append((current_rating-average_other_business)*w)
                            weights.append(w)
                    num1=sum(ratings_diff)
                    den1=sum(abs(each) for each in weights)
                    rating= -1
                    if(num1!=0 and den1!=0):
                        rating=(num1/den1)
                        rating+=average_of_current_business
                        if (rating<1.0):
                            rating=1.0
                        elif (rating>5.0):
                            rating=5.0
                        return (current_user, current_business, str(rating))
                    else:
                        rating = average_of_current_business
                        return (current_user, current_business, str(rating))
                else:
                    return (current_user, current_business, str("3.0"))
            else:
                average_of_current_business = sum_ratings / len(list(current_business_ratings))
                return (current_user, current_business, str(average_of_current_business))

        else:
            return (current_user, current_business, str("3.0"))

    data_raw=sc.textFile(train_file , minPartitions=2)
    first_row=data_raw.first()
    train_data=data_raw.filter(lambda a: a != first_row).map(lambda a: a.split(","))
    test_raw=sc.textFile(test_file).map(lambda x: x.split(","))
    first_row_test=test_raw.first()
    test_data=test_raw.filter(lambda a: a!=first_row_test)
    train_dictionary_1=train_data.map(lambda a: ((a[0]),((a[1]),float(a[2])))).groupByKey().mapValues(dict).collectAsMap()
    train_dictionary_2=train_data.map(lambda a: ((a[1]),((a[0]),float(a[2])))).groupByKey().mapValues(dict).collectAsMap()
    train_dict1=sc.broadcast(train_dictionary_1)
    train_dict2=sc.broadcast(train_dictionary_2)
    predictions=test_data.map(lambda a: predict(train_dict1, train_dict2 ,a[0], a[1]))
    with open(out_file, mode='w') as ofile:
        ofile_w = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ofile_w.writerow(['user_id', ' business_id', ' prediction'])
        if(predictions):
            for i in predictions.collect():
                ofile_w.writerow([i[0], i[1], str(i[2])])
    ofile.close()

def case4(train_file,test_file,out_file):
    def hash_function(each_hash,a):
        return min([((each_hash[0]*x+each_hash[1])%each_hash[2])%m for x in a[1]])

    def j_simi(a):
        a1 = set(characteristics_matrix[bus_num[a[0]]][1])
        a2 = set(characteristics_matrix[bus_num[a[1]]][1])
        intersection = len(a1&a2)
        union = len(a1|a2)
        return (a[0],a[1],intersection/union)

    def sig(x):
        res = []
        for i in range(b):
            res.append(((i, tuple(x[1][i * r:(i + 1) * r])), [x[0]]))
        return res

    def cands(a):
        r=[]
        bus=list(a[1])
        bus.sort()
        for i in range(len(a[1])):
            for j in range(i+1,len(a[1])):
                r.append(((bus[i],bus[j]),1))
        return r

    def predict(train_data, train_dict2_a, current_user, current_business):
        train_dictionary_1 = train_data.value
        train_dictionary_2 = train_dict2_a.value
        if(current_business not in similarPairs1 and current_business not in similarPairs2):
            current_user_ratings = train_dictionary_1.get(current_user)
            sum_ratings = sum(current_user_ratings.values())
            len_ratings=len(list(current_user_ratings))
            avg=sum_ratings/len_ratings
            return (current_user, current_business, str(avg))
        if (train_dictionary_2.get(current_business)):
            current_business_users=list(train_dictionary_2.get(current_business))
            current_business_ratings=train_dictionary_2.get(current_business)
            sum_ratings=sum(current_business_ratings.values())
            ratings_diff=[]
            weights=[]
            if(train_dictionary_1.get(current_user)!=None):
                rated_by_users=list(train_dictionary_1.get(current_user))
                if (rated_by_users):
                    average_of_current_business=sum_ratings/len(list(current_business_ratings))
                    for i in range(len(rated_by_users)):
                        current_business_total_rating=0
                        other_business_total_rating=0
                        k=0
                        c_b_p=[]
                        o_b_p=[]
                        current_rating=train_dictionary_2[rated_by_users[i]].get(current_user)
                        while (k<len(current_business_users)):
                            if (train_dictionary_2[rated_by_users[i]].get(current_business_users[k])):
                                current_business_total_rating+=train_dictionary_2[current_business].get(current_business_users[k])
                                other_business_total_rating+=train_dictionary_2[rated_by_users[i]].get(current_business_users[k])
                                o_b_p.append(train_dictionary_2[rated_by_users[i]].get(current_business_users[k]))
                                c_b_p.append(train_dictionary_2[current_business].get(current_business_users[k]))
                            k+=1
                        k=0
                        if(c_b_p):
                            den_first = 0
                            den_second = 0
                            w = 0
                            num = 0
                            den = 0
                            first = 0
                            second = 0
                            average_other_business=other_business_total_rating/len(o_b_p)
                            averagecurrent_business = current_business_total_rating / len(c_b_p)
                            for i in range(len(c_b_p)):
                                first=c_b_p[i]-averagecurrent_business
                                second=o_b_p[i]-average_other_business
                                num+=(first*second)
                                den_first+=(first*first)
                                den_second+=(second*second)
                            den=sqrt(den_first)*sqrt(den_second)
                            if (den!=0):
                                w = num/den
                            ratings_diff.append((current_rating-average_other_business)*w)
                            weights.append(w)
                    num1=sum(ratings_diff)
                    den1=sum(abs(each) for each in weights)
                    rating= -1
                    if(num1!=0 and den1!=0):
                        rating=(num1/den1)
                        rating+=average_of_current_business
                        if (rating<1.0):
                            rating=1.0
                        elif (rating>5.0):
                            rating=5.0
                        return (current_user, current_business, str(rating))
                    else:
                        rating = average_of_current_business
                        return (current_user, current_business, str(rating))
                else:
                    return (current_user, current_business, str("3.0"))
            else:
                average_of_current_business = sum_ratings / len(list(current_business_ratings))
                return (current_user, current_business, str(average_of_current_business))

        else:
            return (current_user, current_business, str("3.0"))

    data_raw=sc.textFile(train_file, minPartitions=2)
    first_row=data_raw.first()
    data=data_raw.filter(lambda a: a != first_row).map(lambda a: a.split(","))

    b=data.map(lambda a: (a[1],1)).reduceByKey(lambda a,b: a).map(lambda a: a[0])
    business=list(b.collect())
    business.sort()
    usr = data.map(lambda a: (a[0],1)).reduceByKey(lambda a,b: a).map(lambda a: a[0])
    users=list(usr.collect())
    users.sort()
    m=len(users)
    hash_values = [[11, 43, 107], [2, 63, 97], [17, 23, 769], [181, 251, 421], [13, 37, 3079], [14, 91, 1543],
                   [73, 803, 49157], [1, 101, 193], [913, 901, 24593], [91, 29, 12289], [8, 119, 389], [41, 443, 6311],
                   [3, 79, 53], [387, 552, 98317], [71, 67, 6151], [887, 2281, 7817]]
    n = len(hash_values)
    b = 8
    r = int(n / b)
    bus_num={}
    user_num={}
    for num, bus_name in enumerate(business):
        bus_num[bus_name]=num
    for num, user_name in enumerate(users):
        user_num[user_name]=num
    user_dict=sc.broadcast(user_num)
    charmat=data.map(lambda a: (a[1], [user_dict.value[a[0]]])).reduceByKey(lambda a,b: a+b)
    char_matrix=charmat.sortBy(lambda a: a[0])
    characteristics_matrix=char_matrix.collect()
    signatures=char_matrix.map(lambda a: (a[0], [hash_function(each_hash,a) for each_hash in hash_values]))
    candidate1=signatures.flatMap(sig).reduceByKey(lambda a,b: a+b).filter(lambda a: len(a[1])>1)
    candidate=candidate1.flatMap(cands).reduceByKey(lambda a,b: a).map(lambda a: a[0])
    result = candidate.map(j_simi).filter(lambda a: a[2]>=0.5).sortBy(lambda a: a[1]).sortBy(lambda a: a[0])
    result1 = result.collect()

    similarPairs1 = sc.parallelize(result1).map(lambda x : (x[0], x[1])).groupByKey().sortByKey().mapValues(list).collectAsMap()
    similarPairs2 = sc.parallelize(result1).map(lambda x : (x[1], x[0])).groupByKey().sortByKey().mapValues(list).collectAsMap()
    train_dictionary_1 = data.map(lambda a: ((a[0]), ((a[1]), float(a[2])))).groupByKey().mapValues(dict).collectAsMap()
    train_dictionary_2 = data.map(lambda a: ((a[1]), ((a[0]), float(a[2])))).groupByKey().mapValues(dict).collectAsMap()
    train_dict1 = sc.broadcast(train_dictionary_1)
    train_dict2 = sc.broadcast(train_dictionary_2)
    test_raw=sc.textFile(test_file).map(lambda x: x.split(","))
    first_row_test=test_raw.first()
    test_data=test_raw.filter(lambda a: a!=first_row_test)
    predictions=test_data.map(lambda a: predict(train_dict1, train_dict2 ,a[0], a[1]))
    with open(out_file, mode='w') as ofile:
        ofile_w = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ofile_w.writerow(['user_id', ' business_id', ' prediction'])
        if (predictions):
            for i in predictions.collect():
                ofile_w.writerow([i[0], i[1], str(i[2])])
    ofile.close()
    
sc = SparkContext('local[*]', 'c1')
case=int(sys.argv[3])
train_file=sys.argv[1]
test_file = sys.argv[2]
out_file = sys.argv[4]

if(case==1):
    case1(train_file,test_file,out_file)
elif(case==2):
    case2(train_file,test_file,out_file)
elif(case==3):
    case3(train_file,test_file,out_file)
else :
    case4(train_file,test_file,out_file)
    
print("Time: ",time.time()-s)
