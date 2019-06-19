from pyspark import SparkContext
from math import sqrt
import csv
import sys
import time
from pyspark.mllib.recommendation import ALS, Rating
import os
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
start_time = time.time()


'''-------------------------- Model-based CF recommendation system with Spark MLlib --------------------------'''


def model_based(train_file, test_file, out_file):
    def find(x):
        if (x[0] not in train_user_num):
            user = test_user_num[x[0]]
        else:
            user = train_user_num[x[0]]
        if (x[1] not in train_bus_num):
            busines = test_bus_num[x[1]]
        else:
            busines = train_bus_num[x[1]]

        return (user, busines)

    def rates(a):
        if (a[2] < 1):
            rate = 1.0
        elif (a[2] > 5):
            rate = 5.0
        else:
            rate = a[2]
        return ((a[0], a[1]), rate)

    # Read training file
    data_raw = sc.textFile(train_file)

    # Remove the header from training data
    first_row = data_raw.first()
    data1 = data_raw.filter(lambda a: a != first_row).map(lambda a: a.split(","))

    # Read test file
    test_raw = sc.textFile(test_file).map(lambda x: x.split(","))

    # Remove the header from test data
    first_row_test = test_raw.first()
    test_data = test_raw.filter(lambda a: a != first_row_test)

    # Get list of businesses for training data
    b = data1.map(lambda a: (a[1], 1)).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    business = list(b.collect())

    # Get list of users for training data
    usr = data1.map(lambda a: (a[0], 1)).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    users = list(usr.collect())

    # Create 2 way dictionaries for users and businesses in training data
    train_bus_num = {}
    train_user_num = {}
    train_bus_num1 = {}
    train_user_num1 = {}
    for num, bus_name in enumerate(business):
        train_bus_num[bus_name] = num
        train_bus_num1[num] = bus_name
    for num, user_name in enumerate(users):
        train_user_num[user_name] = num
        train_user_num1[num] = user_name

    total_users = len(train_user_num)
    total_business = len(train_bus_num)

    data = data1.map(lambda a: ((train_user_num[a[0]], train_bus_num[a[1]]), float(a[2])))

    # Get list of businesses for test data
    tb = test_data.map(lambda a: (a[1], 1)).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    tbusiness = list(tb.collect())

    # Get list of users for test data
    tusr = test_data.map(lambda a: (a[0], 1)).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    tusers = list(tusr.collect())

    # Create 2 way dictionaries for users and businesses in test data
    test_user_num = {}
    test_user_num1 = {}
    test_bus_num1 = {}
    test_bus_num = {}
    for num, bus_name in enumerate(tbusiness):
        if (bus_name not in train_bus_num):
            test_bus_num[bus_name] = num + total_business + 1
            test_bus_num1[num + total_business + 1] = bus_name
    for num, user_name in enumerate(tusers):
        if (user_name not in train_user_num):
            test_user_num[user_name] = num + total_users + 1
            test_user_num1[num + total_users + 1] = user_name

    test_map = test_data.map(find)  # check if test user/business is in training data or not
    test_data = test_map.map(lambda x: (x, None))
    ratings = data.map(lambda a: Rating(a[0][0], a[0][1], a[1]))

    # Paramaters for ALS
    numIterations = 10
    rank = 4

    # train the model
    model = ALS.train(ratings, rank, numIterations)
    # Predict ratings for test data
    predic = model.predictAll(test_map).map(rates)
    predic_new = test_data.subtractByKey(predic).map(lambda a: (a[0], 3.0))
    predictions = sc.union([predic, predic_new])

    # Write predictions to output csv file
    with open(out_file, mode='w') as ofile:
        ofile_w = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ofile_w.writerow(['user_id', ' business_id', ' prediction'])
        if (predictions):
            for e in predictions.collect():
                if (e[0][0] not in train_user_num1 and e[0][1] not in train_bus_num1):
                    ofile_w.writerow([test_user_num1[e[0][0]], test_bus_num1[e[0][1]], e[1]])
                elif (e[0][0] not in train_user_num1):
                    ofile_w.writerow([test_user_num1[e[0][0]], train_bus_num1[e[0][1]], e[1]])
                elif (e[0][1] not in train_bus_num1):
                    ofile_w.writerow([train_user_num1[e[0][0]], test_bus_num1[e[0][1]], e[1]])
                else:
                    ofile_w.writerow([train_user_num1[e[0][0]], train_bus_num1[e[0][1]], e[1]])
    ofile.close()


'''-------------------------- User-based CF recommendation system --------------------------'''


def user_based(train_file, test_file, out_file):
    def predict(train_data, train_dict2_a, current_user, current_business):
        train_dictionary_1 = train_data.value
        train_dict2 = train_dict2_a.value

        # If current test user is in users of training data
        if (train_dictionary_1.get(current_user)):
            current_user_businesses = list(train_dictionary_1.get(current_user))
            current_user_ratings = train_dictionary_1.get(current_user)
            sum_ratings = sum(current_user_ratings.values())
            ratings_diff = []
            weights = []

            # If current test business is in training data
            if (train_dict2.get(current_business) != None):
                rated_by_users = list(train_dict2.get(current_business))

                # If current business has been rated by any user
                if (rated_by_users):
                    average_of_current_user = sum_ratings / len(list(current_user_ratings))
                    for i in range(len(rated_by_users)):
                        current_user_total_rating = 0
                        other_user_total_rating = 0
                        k = 0
                        c_b_p = []
                        o_b_p = []
                        current_rating = train_dictionary_1[rated_by_users[i]].get(current_business)
                        while (k < len(current_user_businesses)):
                            if (train_dictionary_1[rated_by_users[i]].get(current_user_businesses[k])):
                                current_user_total_rating += train_dictionary_1[current_user].get(
                                    current_user_businesses[k])
                                other_user_total_rating += train_dictionary_1[rated_by_users[i]].get(
                                    current_user_businesses[k])
                                o_b_p.append(train_dictionary_1[rated_by_users[i]].get(current_user_businesses[k]))
                                c_b_p.append(train_dictionary_1[current_user].get(current_user_businesses[k]))
                            k += 1
                        k = 0
                        if (c_b_p):

                            # Calculate similarity
                            den_first = 0
                            den_second = 0
                            w = 0
                            num = 0
                            den = 0
                            first = 0
                            second = 0
                            average_other_user = other_user_total_rating / len(o_b_p)
                            averagecurrent_user = current_user_total_rating / len(c_b_p)
                            for i in range(len(c_b_p)):
                                first = c_b_p[i] - averagecurrent_user
                                second = o_b_p[i] - average_other_user
                                num += (first * second)
                                den_first += (first * first)
                                den_second += (second * second)
                            den = sqrt(den_first) * sqrt(den_second)
                            if (den != 0):
                                w = num / den
                            ratings_diff.append((current_rating - average_other_user) * w)
                            weights.append(w)
                    num1 = sum(ratings_diff)
                    den1 = sum(abs(each) for each in weights)
                    rating = -1
                    if (num1 != 0 and den1 != 0):
                        rating = (num1 / den1)
                        rating += average_of_current_user
                        if (rating < 1.0):
                            rating = 1.0
                        elif (rating > 5.0):
                            rating = 5.0
                        return (current_user, current_business, str(rating))
                    else:
                        rating = average_of_current_user
                        return (current_user, current_business, str(rating))

                # If current business has NOT been rated by any user, predict rating = 3
                else:
                    return (current_user, current_business, str("3.0"))

            # If current test business is NOT in training data, predict avg of current test user
            else:
                average_of_current_user = sum_ratings / len(list(current_user_ratings))
                return (current_user, current_business, str(average_of_current_user))

        # If current test user is NOT in users of training data, predict rating = 3
        else:
            return (current_user, current_business, str("3.0"))

    # Read training file
    data_raw = sc.textFile(train_file, minPartitions=2)
    # Remove the header from training data
    first_row = data_raw.first()
    train_data = data_raw.filter(lambda a: a != first_row).map(lambda a: a.split(","))

    # Read test file
    test_raw = sc.textFile(test_file).map(lambda x: x.split(","))
    # Remove the header from test data
    first_row_test = test_raw.first()
    test_data = test_raw.filter(lambda a: a != first_row_test)

    # Create a dictionary Key : user Value: corresponding businesses
    train_dictionary_1 = train_data.map(lambda a: ((a[0]), ((a[1]), float(a[2])))).groupByKey().mapValues(
        dict).collectAsMap()
    # Create a dictionary Key : business Value: corresponding users
    train_dictionary_2 = train_data.map(lambda a: ((a[1]), ((a[0]), float(a[2])))).groupByKey().mapValues(
        dict).collectAsMap()
    train_dict1 = sc.broadcast(train_dictionary_1)
    train_dict2 = sc.broadcast(train_dictionary_2)

    # Generate predictions for test data
    predictions = test_data.map(lambda a: predict(train_dict1, train_dict2, a[0], a[1]))

    # Write predictions to output csv file
    with open(out_file, mode='w') as ofile:
        ofile_w = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ofile_w.writerow(['user_id', ' business_id', ' prediction'])
        if (predictions):
            for i in predictions.collect():
                ofile_w.writerow([i[0], i[1], str(i[2])])
    ofile.close()


'''--------------------------  Item-based CF recommendation system --------------------------'''

def item_based(train_file, test_file, out_file):
    def predict(train_data, train_dict2_a, current_user, current_business):
        train_dictionary_1 = train_data.value
        train_dictionary_2 = train_dict2_a.value

        # If current test business is in training data
        if (train_dictionary_2.get(current_business)):
            current_business_users = list(train_dictionary_2.get(current_business))
            current_business_ratings = train_dictionary_2.get(current_business)
            sum_ratings = sum(current_business_ratings.values())
            ratings_diff = []
            weights = []

            # If current test user is in users of training data
            if (train_dictionary_1.get(current_user) != None):
                rated_by_users = list(train_dictionary_1.get(current_user))

                # If current user has rated any business
                if (rated_by_users):
                    average_of_current_business = sum_ratings / len(list(current_business_ratings))
                    for i in range(len(rated_by_users)):
                        current_business_total_rating = 0
                        other_business_total_rating = 0
                        k = 0
                        c_b_p = []
                        o_b_p = []
                        current_rating = train_dictionary_2[rated_by_users[i]].get(current_user)
                        while (k < len(current_business_users)):
                            if (train_dictionary_2[rated_by_users[i]].get(current_business_users[k])):
                                current_business_total_rating += train_dictionary_2[current_business].get(
                                    current_business_users[k])
                                other_business_total_rating += train_dictionary_2[rated_by_users[i]].get(
                                    current_business_users[k])
                                o_b_p.append(train_dictionary_2[rated_by_users[i]].get(current_business_users[k]))
                                c_b_p.append(train_dictionary_2[current_business].get(current_business_users[k]))
                            k += 1
                        k = 0
                        if (c_b_p):
                            # Calculate similarity
                            den_first = 0
                            den_second = 0
                            w = 0
                            num = 0
                            den = 0
                            first = 0
                            second = 0
                            average_other_business = other_business_total_rating / len(o_b_p)
                            averagecurrent_business = current_business_total_rating / len(c_b_p)
                            for i in range(len(c_b_p)):
                                first = c_b_p[i] - averagecurrent_business
                                second = o_b_p[i] - average_other_business
                                num += (first * second)
                                den_first += (first * first)
                                den_second += (second * second)
                            den = sqrt(den_first) * sqrt(den_second)
                            if (den != 0):
                                w = num / den
                            ratings_diff.append((current_rating - average_other_business) * w)
                            weights.append(w)
                    num1 = sum(ratings_diff)
                    den1 = sum(abs(each) for each in weights)
                    rating = -1
                    if (num1 != 0 and den1 != 0):
                        rating = (num1 / den1)
                        rating += average_of_current_business
                        if (rating < 1.0):
                            rating = 1.0
                        elif (rating > 5.0):
                            rating = 5.0
                        return (current_user, current_business, str(rating))
                    else:
                        rating = average_of_current_business
                        return (current_user, current_business, str(rating))

                # If current user has NOT rated any business, predict rating = 3
                else:
                    return (current_user, current_business, str("3.0"))

            # If current test user is NOT in users of training data, predict avg of current test business
            else:
                average_of_current_business = sum_ratings / len(list(current_business_ratings))
                return (current_user, current_business, str(average_of_current_business))

        # If current test business is NOT in training data, predict rating = 3
        else:
            return (current_user, current_business, str("3.0"))
    # Read training file
    data_raw = sc.textFile(train_file, minPartitions=2)
    # Remove the header from training data
    first_row = data_raw.first()
    train_data = data_raw.filter(lambda a: a != first_row).map(lambda a: a.split(","))

    # Read test file
    test_raw = sc.textFile(test_file).map(lambda x: x.split(","))
    # Remove the header from test data
    first_row_test = test_raw.first()
    test_data = test_raw.filter(lambda a: a != first_row_test)

    # Create a dictionary Key : user Value: corresponding businesses
    train_dictionary_1 = train_data.map(lambda a: ((a[0]), ((a[1]), float(a[2])))).groupByKey().mapValues(
        dict).collectAsMap()
    # Create a dictionary Key : business Value: corresponding users
    train_dictionary_2 = train_data.map(lambda a: ((a[1]), ((a[0]), float(a[2])))).groupByKey().mapValues(
        dict).collectAsMap()
    train_dict1 = sc.broadcast(train_dictionary_1)
    train_dict2 = sc.broadcast(train_dictionary_2)

    # Generate predictions for test data
    predictions = test_data.map(lambda a: predict(train_dict1, train_dict2, a[0], a[1]))

    # Write predictions to output csv file
    with open(out_file, mode='w') as ofile:
        ofile_w = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ofile_w.writerow(['user_id', ' business_id', ' prediction'])
        if (predictions):
            for i in predictions.collect():
                ofile_w.writerow([i[0], i[1], str(i[2])])
    ofile.close()


'''--------------------------  Item-based CF recommendation system with Jaccard based LSH --------------------------'''

def item_based_using_LSH(train_file, test_file, out_file):
    def hash_function(each_hash, a):
        return min([((each_hash[0] * x + each_hash[1]) % each_hash[2]) % m for x in a[1]])

    def j_simi(a): # Calculate Jaccard Similarity
        a1 = set(characteristics_matrix[bus_num[a[0]]][1])
        a2 = set(characteristics_matrix[bus_num[a[1]]][1])
        intersection = len(a1 & a2)
        union = len(a1 | a2)
        return (a[0], a[1], intersection / union)

    def sig(x): # Generate signatures
        res = []
        for i in range(b):
            res.append(((i, tuple(x[1][i * r:(i + 1) * r])), [x[0]]))
        return res

    def cands(a): # Generate candidates
        r = []
        bus = list(a[1])
        bus.sort()
        for i in range(len(a[1])):
            for j in range(i + 1, len(a[1])):
                r.append(((bus[i], bus[j]), 1))
        return r

    def predict(train_data, train_dict2_a, current_user, current_business):
        train_dictionary_1 = train_data.value
        train_dictionary_2 = train_dict2_a.value

        # If current business is not in similar pairs, predict average rating of current user
        if (current_business not in similarPairs1 and current_business not in similarPairs2):
            current_user_ratings = train_dictionary_1.get(current_user)
            sum_ratings = sum(current_user_ratings.values())
            len_ratings = len(list(current_user_ratings))
            avg = sum_ratings / len_ratings
            return (current_user, current_business, str(avg))

        # If current test business is in business of training data
        if (train_dictionary_2.get(current_business)):
            current_business_users = list(train_dictionary_2.get(current_business))
            current_business_ratings = train_dictionary_2.get(current_business)
            sum_ratings = sum(current_business_ratings.values())
            ratings_diff = []
            weights = []

            # If current test user is in users of training data
            if (train_dictionary_1.get(current_user) != None):
                rated_by_users = list(train_dictionary_1.get(current_user))

                # If current user has rated any business
                if (rated_by_users):
                    average_of_current_business = sum_ratings / len(list(current_business_ratings))
                    for i in range(len(rated_by_users)):
                        current_business_total_rating = 0
                        other_business_total_rating = 0
                        k = 0
                        c_b_p = []
                        o_b_p = []
                        current_rating = train_dictionary_2[rated_by_users[i]].get(current_user)
                        while (k < len(current_business_users)):
                            if (train_dictionary_2[rated_by_users[i]].get(current_business_users[k])):
                                current_business_total_rating += train_dictionary_2[current_business].get(
                                    current_business_users[k])
                                other_business_total_rating += train_dictionary_2[rated_by_users[i]].get(
                                    current_business_users[k])
                                o_b_p.append(train_dictionary_2[rated_by_users[i]].get(current_business_users[k]))
                                c_b_p.append(train_dictionary_2[current_business].get(current_business_users[k]))
                            k += 1
                        k = 0
                        if (c_b_p):
                            # Calculate similarity
                            den_first = 0
                            den_second = 0
                            w = 0
                            num = 0
                            den = 0
                            first = 0
                            second = 0
                            average_other_business = other_business_total_rating / len(o_b_p)
                            averagecurrent_business = current_business_total_rating / len(c_b_p)
                            for i in range(len(c_b_p)):
                                first = c_b_p[i] - averagecurrent_business
                                second = o_b_p[i] - average_other_business
                                num += (first * second)
                                den_first += (first * first)
                                den_second += (second * second)
                            den = sqrt(den_first) * sqrt(den_second)
                            if (den != 0):
                                w = num / den
                            ratings_diff.append((current_rating - average_other_business) * w)
                            weights.append(w)
                    num1 = sum(ratings_diff)
                    den1 = sum(abs(each) for each in weights)
                    rating = -1
                    if (num1 != 0 and den1 != 0):
                        rating = (num1 / den1)
                        rating += average_of_current_business
                        if (rating < 1.0):
                            rating = 1.0
                        elif (rating > 5.0):
                            rating = 5.0
                        return (current_user, current_business, str(rating))
                    else:
                        rating = average_of_current_business
                        return (current_user, current_business, str(rating))

                # If current user has NOT rated any business, predict rating = 3
                else:
                    return (current_user, current_business, str("3.0"))

            # If current test user is NOT in users of training data, predict avg of current test business
            else:
                average_of_current_business = sum_ratings / len(list(current_business_ratings))
                return (current_user, current_business, str(average_of_current_business))

        # If current test business is NOT in training data, predict rating = 3
        else:
            return (current_user, current_business, str("3.0"))


    # Read training file
    data_raw = sc.textFile(train_file, minPartitions=2)
    # Remove the header from training data
    first_row = data_raw.first()
    data = data_raw.filter(lambda a: a != first_row).map(lambda a: a.split(","))

    # Get list of businesses for training data
    b = data.map(lambda a: (a[1], 1)).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    business = list(b.collect())
    business.sort()

    # Get list of users for training data
    usr = data.map(lambda a: (a[0], 1)).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    users = list(usr.collect())
    users.sort()

    # m,n,b,r are parameters for the algotithm
    m = len(users)
    hash_values = [[11, 43, 107], [2, 63, 97], [17, 23, 769], [181, 251, 421], [13, 37, 3079], [14, 91, 1543],
                   [73, 803, 49157], [1, 101, 193], [913, 901, 24593], [91, 29, 12289], [8, 119, 389], [41, 443, 6311],
                   [3, 79, 53], [387, 552, 98317], [71, 67, 6151], [887, 2281, 7817]]
    n = len(hash_values)
    b = 8
    r = int(n / b)
    bus_num = {}
    user_num = {}
    for num, bus_name in enumerate(business):
        bus_num[bus_name] = num
    for num, user_name in enumerate(users):
        user_num[user_name] = num
    user_dict = sc.broadcast(user_num)

    # Create characteristics matrix
    charmat = data.map(lambda a: (a[1], [user_dict.value[a[0]]])).reduceByKey(lambda a, b: a + b)
    char_matrix = charmat.sortBy(lambda a: a[0])
    characteristics_matrix = char_matrix.collect()
    # Generate signatures
    signatures = char_matrix.map(lambda a: (a[0], [hash_function(each_hash, a) for each_hash in hash_values]))
    # Generate candidates
    candidate1 = signatures.flatMap(sig).reduceByKey(lambda a, b: a + b).filter(lambda a: len(a[1]) > 1)
    candidate = candidate1.flatMap(cands).reduceByKey(lambda a, b: a).map(lambda a: a[0])
    result = candidate.map(j_simi).filter(lambda a: a[2] >= 0.5).sortBy(lambda a: a[1]).sortBy(lambda a: a[0])
    result1 = result.collect()

    # Create 2 way similar business pairs
    similarPairs1 = sc.parallelize(result1).map(lambda x: (x[0], x[1])).groupByKey().sortByKey().mapValues(
        list).collectAsMap()
    similarPairs2 = sc.parallelize(result1).map(lambda x: (x[1], x[0])).groupByKey().sortByKey().mapValues(
        list).collectAsMap()
    train_dictionary_1 = data.map(lambda a: ((a[0]), ((a[1]), float(a[2])))).groupByKey().mapValues(dict).collectAsMap()
    train_dictionary_2 = data.map(lambda a: ((a[1]), ((a[0]), float(a[2])))).groupByKey().mapValues(dict).collectAsMap()
    train_dict1 = sc.broadcast(train_dictionary_1)
    train_dict2 = sc.broadcast(train_dictionary_2)

    # Read test file
    test_raw = sc.textFile(test_file).map(lambda x: x.split(","))
    # Remove the header from test data
    first_row_test = test_raw.first()
    test_data = test_raw.filter(lambda a: a != first_row_test)

    # Generate predictions for test data
    predictions = test_data.map(lambda a: predict(train_dict1, train_dict2, a[0], a[1]))

    # Write predictions to output csv file
    with open(out_file, mode='w') as ofile:
        ofile_w = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ofile_w.writerow(['user_id', ' business_id', ' prediction'])
        if (predictions):
            for i in predictions.collect():
                ofile_w.writerow([i[0], i[1], str(i[2])])
    ofile.close()


sc = SparkContext('local[*]', 'c1')
case = int(sys.argv[3])
train_file = sys.argv[1]
test_file = sys.argv[2]
out_file = sys.argv[4]

if (case == 1):  
    model_based(train_file, test_file, out_file)
elif (case == 2):
    user_based(train_file, test_file, out_file)
elif (case == 3):
    item_based(train_file, test_file, out_file)
else:
    item_based_using_LSH(train_file, test_file, out_file)

print("Time: ", time.time() - start_time)

