import json

cut1 = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
cut2 = [5e-9, 5e-8, 5e-7, 5e-6, 5e-5, 5e-4]
cut3 = [1e-2, 5e-2, 1e-3, 5e-3, 1e-4]

#cut = [0] + cut1 + cut2
cut = cut1 + cut2 + cut3

for n in range(4,8):
    data_dict = {}
    with open('h2_{}_t2.dat'.format(n)) as infile:
    #with open('h2_{}_mp2.dat'.format(n)) as infile:
        i = 0
        for line in infile:
            trash, trash1, val = line.split()
            # For getting energy diffs for MP2
            '''if i == 0:
                data_dict[cut[i]] = val
            else:
                #print(data_dict.keys())
                data_dict[cut[i]] = float(data_dict[0]) - float(val)'''
            data_dict[cut[i]] = float(val)
            i += 1
    with open('h2_{}_t2.json'.format(n), "w") as outfile:
        #with open('h2_{}_mp2.json'.format(n), "w") as outfile:
        json.dump(data_dict, outfile, indent=4)
