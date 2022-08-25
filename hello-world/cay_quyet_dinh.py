from sklearn import tree

#step 1: Thu thap du lieu
#step 2: Xu ly du lieu
#step 3: Xay dung model
#step 4: Du doan ket qua
#step 5: Danh gia ket qua

# Construct tree
my_tree = tree.DecisionTreeClassifier()

dac_trung_origin = [
    ['nhe', 'tb', 'tb', 'nhieu'],
    ['nang', 'thap', 'cao', 'it'],
    ['nhe', 'thap', 'cao', 'it'],
    ['nang', 'cao', 'cao', 'tb'],
    ['nhe', 'cao', 'cao', 'nhieu'],
    ['tb', 'thap', 'tb', 'nhieu'],
    ['tb', 'tb', 'tb', 'it'],
    ['nang', 'thap', 'thap', 'nhieu'],
]
matrix_dac_trung = [
    ['nhe', 1],
    ['thap', 2],
    ['tb', 3],
    ['cao', 4],
    ['nang', 5],
    ['it', 6],
    ['nhieu', 7]
]
matrix_result = [
    ['co', 1],
    ['khong', 0]
]
dac_trung = [
    [1, 3, 3, 7],
    [5, 2, 4, 6],
    [1, 2, 4, 6],
    [5, 4, 4, 3],
    [1, 4, 4, 7],
    [3, 2, 3, 7],
    [3, 3, 3, 6],
    [5, 2, 3, 7],
]
nhan = [0, 1, 1, 0, 0, 0, 0, 1]

# training
result = my_tree.fit(dac_trung, nhan)

kq = result.predict([[1, 4, 3, 6]])
kq2 = result.predict([[1, 4, 3, 7]])
kq3 = result.predict([[5, 2, 4, 4]])

print(1, kq)
print(2, kq2)
print(3, kq3)