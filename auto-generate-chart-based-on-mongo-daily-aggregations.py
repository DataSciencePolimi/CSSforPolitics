import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import defaultdict

# the output classes: 0 neutral, 1 remain, 2 leave, 3 mixed
# this algorithm is able to generate automatically a stacked bar chart from the output of mongo query, without doing any modification on the output..
# which mongo query?
# --> db.tweet.aggregate([{"$group" : {_id : {mydate: {$substr: [ "$datetime", 0,10]}, p1:'$p1'}, total:{$sum :1}}},{"$project" : {mydate: '$_id.mydate', p1: '$_id.p1', total: '$total', _id : 0}},  { $out : "myp6" }  ], { allowDiskUse: true })

try:

    file = open("C:/Users/emre2/Desktop/results_daily.csv", "r")
    remains = []
    leaves = []
    mixes = []
    neutrals = []

    counter = 0
    mylist = file.read().splitlines()
    filteredlist = []

    # filter for the dates you want to visualize
    for line in mylist:
        if '2016-06' in line:
            cols = line.split(",")
            if len(cols) != 4:
                continue;
            filteredlist.append(line)

    #create a nested dictionary to keep the data in correct format
    mydict = {}
    counter = 0
    for line in filteredlist:
        counter = counter + 1
        cols = line.split(",")
        date1 = cols[1]
        date1split = date1.split(":")
        datekey = date1split[1]
        datekey = datekey.replace("\"","")

        p1 = cols[2]
        p1split = p1.split(":")
        p1value = p1split[1]
        p1value = p1value.replace("\"","")
        p1value = "p" + p1value

        total = cols[3]
        totalsplit = total.split(":")
        totalvalue = totalsplit[1]
        totalvalue = totalvalue.replace("}","")
        totalvalue = totalvalue.replace(".0", "")
        totalvalue = int(totalvalue)

        valueasdict = {}

        valueasdict[p1value] = totalvalue

        for key, value in mydict.items():
            if key == datekey:
                for key2, value2 in value.items():
                    valueasdict[key2]=value2

        mydict[datekey] = valueasdict

        print("done")

    #check missing data for each opinion. if there is, create and set its value to 0
    for key, value in mydict.items():
        existp0 = False
        existp1 = False
        existp2 = False
        existp3 = False
        for key2, value2 in value.items():
                if key2 == "p0":
                    existp0 = True
                elif key2 == "p1":
                    existp1 = True
                elif key2 == "p2":
                    existp2 = True
                elif key2 == "p3":
                    existp3 = True

        if not existp0:
            print("alert, missing data p0 for date: " + key)
            value["p0"] = 0
        if not existp1:
            print("alert, missing data p1 for date: " + key)
            value["p1"] = 0
        if not existp2:
            print("alert, missing data p2 for date: " + key)
            value["p2"] = 0
        if not existp3:
            print("alert, missing data p3 for date: " + key)
            value["p3"]=0
        print("done")

    # order nested dictionary by date column ascending
    od = collections.OrderedDict(sorted(mydict.items()))

    # initialize stacked bar chart parameters

    # width describe the width of bars
    width = 0.35

    # horizontal_dates describe the horizontal axis, dates
    horizontal_dates = list(od.keys())
    y_offset = np.zeros(len(horizontal_dates))

    remains = []
    leaves = []
    mixes = []
    neutrals = []

    # create data for each opinion in each column
    for key,value in od.items():
        for nestedkey, nestedvalue in value.items():
            if(nestedkey=="p0"):
                neutrals.append(nestedvalue)
            elif(nestedkey=="p1"):
                remains.append(nestedvalue)
            elif(nestedkey=="p2"):
                leaves.append(nestedvalue)
            elif (nestedkey == "p3"):
                mixes.append(nestedvalue)

    # convert to numpy array
    remains =  np.array(remains)
    leaves = np.array(leaves)
    mixes = np.array(mixes)
    neutrals = np.array(neutrals)

    if remains.size != leaves.size or leaves.size != mixes.size or mixes.size != neutrals.size:
        print("error, the size of lists that identify opinions should be equal")
        exit(-1)

    # when we visualize also neutrals, it becomes difficult to see in detail,
    # because the data is relatively higher than other opinions
    # so we have a flag that enables/disables this output
    isneutralenabled = False

    N = len(leaves)

    ind = np.arange(N)

    p_remains = plt.bar(ind, remains, width)
    p_leaves = plt.bar(ind, leaves, width, bottom=remains)
    p_mixes = plt.bar(ind, mixes, width, bottom=remains+leaves)

    if isneutralenabled:
        p_neutrals = plt.bar(ind, neutrals, width, bottom=remains+leaves+mixes)
        displaymaxlimit = neutrals.max() + leaves.max() + mixes.max() + remains.max()
    else:
        displaymaxlimit = leaves.max() + mixes.max() + remains.max()

    displayinterval = int(displaymaxlimit/100)

    plt.ylabel('Values')
    plt.title('Distribution of opinions - 2016')
    plt.xticks(ind, columns)
    plt.yticks(np.arange(0, displaymaxlimit, displayinterval))

    if isneutralenabled:
        plt.legend((p_remains[0], p_leaves[0], p_mixes[0], p_neutrals[0]), ('Remains', 'Leaves', 'Mixes','Neutrals'))
    else:
        plt.legend((p_remains[0], p_leaves[0], p_mixes[0]), ('Remains', 'Leaves', 'Mixes'))

    # create and dispay the chart
    plt.show(block=True)

except Exception as exception:
    print('Oops!  An error occurred.  Try again...', exception)