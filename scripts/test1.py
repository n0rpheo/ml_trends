
dict1 = dict()

dict1[0] = set('abc')
dict1[1] = set('abc')

dict2 = dict()
for k in range(0, len(dict1)):
    dict2[k] = dict1[k].copy()

dict1[0].add('d')

symdiff = dict2[0].symmetric_difference(dict1[0])

print(symdiff)

if len(symdiff) == 0:
    print("same")
else:
    print("not same")
