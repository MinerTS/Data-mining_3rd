import math

with open('作業/第三次/glass.txt', 'r') as file:
    lines = file.readlines()
# 自定義column name
column_names = ['Id number', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','Type of glass']
Attributes = [0,1,2,3,4,5,6,7,8]
# 初始化字典
raw_data = []
Ground_tru = []

for line in lines:
    row = line.strip().split(',')
    Ground_tru.append(row.pop(10))
    if len(row) == len(Attributes)+1:
            instance = dict(zip(Attributes, row[1:]))
            raw_data.append(instance)

def equalfrequency(Class, class_index):
    # 設定每個interval的instances#
    interval_freq = len(Class) // 10
    bin_sizes = [interval_freq] * 10
    Sorted_class = sorted(Class)

    # 於平均分配前四個
    remainder = len(Class) % 10
    for i in range(remainder):
        bin_sizes[i] += 1

    # 計算SplittingPoint
    Splitting_point = [0]
    for size in bin_sizes:
        next_point = Splitting_point[-1] + size
        Splitting_point.append(next_point)

    Splitting_intervals = [0] #紀錄intervals
    # for i in Splitting_point[:10]:
    #     Splitting_intervals.append(Sorted_class[i])
    # Splitting_intervals.append(Sorted_class[213])
   

    discretized_class = Class
    points_position = {}
    
    for i in range(len(Class)):
        points_position[i] = Class[i]
    Sorted_class = sorted(points_position.items(), key=lambda x: x[1])
    for point in Splitting_point[1:10]:
        Splitting_value = (Sorted_class[point - 1][1] + Sorted_class[point][1]) / 2
        Splitting_intervals.append(Splitting_value)
    Splitting_intervals.append(max(Class))

    intervals_num = []
    for i in range(0,10):
        intervals = []
        for j in range(Splitting_point[i], Splitting_point[i+1]):
            intervals.append(Sorted_class[j][0])
        intervals_num.append(intervals)
    
    for i in range(10):
        label = intervals_num[i]
        for j in label:
            discretized_class[j] = i + 1



    # for value in Class:
    #     for j in range(10):
    #         if Splitting_point[j] <= Sorted_class.index(value) < Splitting_point[j+1]:
    #              discretized_class.append(j + 1)
    #         if value == max(Class):
    #             discretized_class.append(10)
    #             break

    Splitting_str = ["{:.4f}".format(num) for num in Splitting_intervals] # 調整浮點數顯示位數
    class_name = column_names[Attributes[class_index] + 1]  # match column name
    print(f"Splitting Points(include Max. and Min.) for {class_name}:\n{Splitting_str}")

    return discretized_class

print("\nRun Equal Frequency:\n")

discretized_row = []
discr_freq_data = []  #EquFreq後的資料，List包含Dict裡面的value為str

# 循环处理每个属性
for class_selected in Attributes:  # 跳过第一个和最后一个列名
    valuesTodis = [float(row[class_selected]) for row in raw_data]
    discr_freq_values = equalfrequency(valuesTodis, class_selected)  #執行EquFreq

    discr_freq_str = []
    for value in discr_freq_values:
        str_value = str(value)
        discr_freq_str.append(str_value)

    discretized_row.append(discr_freq_str)

for values in zip(*discretized_row):
    discretized_dict = {key: value for key, value in zip(Attributes, values)}
    discr_freq_data.append(discretized_dict)

def probability(data):   #計算類別資料發生率公式
    
    total_samples = len(data)
    value_counts = {}
   
    for value in data:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    prob = {key: value / total_samples for key, value in value_counts.items()}

    return prob

def H_entropy(data):   # 計算Entropye公式_H(X),H(Y)
    probs = probability(data)
    probs_values = [prob[1] for prob in probs.items()]


    entropy_value = -sum(p * math.log2(p) for p in probs_values)
    return entropy_value

def Hxy_entropy(class1, class2):   #計算Entropye公式_H(X,Y)
    Hxy_data = list(zip(class1, class2))
    Hxy_prob = probability(Hxy_data)
    Hxy_entropy = -sum(p * math.log2(p) for p in Hxy_prob.values())
    return Hxy_entropy

def sym_uncert(p_x, p_y):   #計算symmetric_uncertainty公式_U(X,Y)
    h_x = H_entropy(p_x)
    h_y = H_entropy(p_y)
    h_xy = Hxy_entropy(p_x,p_y)

    Uxy = 2 * (h_x + h_y - h_xy) / (h_x + h_y)
    return Uxy

def sym_uncertC(p_x):   #計算symmetric_uncertainty公式_U(X,C)
    h_x = H_entropy(p_x)
    h_C = H_entropy(Ground_tru)
    h_xC = Hxy_entropy(p_x,Ground_tru)

    Uxy = 2 * (h_x + h_C - h_xC) / (h_x + h_C)
    return Uxy

def Goodness(class1,selected_f):
    num = 0   #分子計數
    den = 0   #分母計數
    
    # 計算分子：對已選特徵套用sym_uncertC函數的結果相加
    for feature_index in range(len(selected_f)):
        feature = Attributes.index(feature_index)
        num += sym_uncertC(class1[feature])

    # 計算分母：對已選特徵兩兩組合套用sym_uncert函數的結果相加
    if len(selected_f) ==1:
        den += 1
    else:
        for i in range(len(selected_f)):
            for j in range(len(selected_f)):
                den += sym_uncert(class1[i], class1[j])

    if den == 0:
        Gn = 0
    else:
        Gn = num / math.sqrt(den)
    return Gn

# 初始化已選Attributes和Goodness值
fwd_selected_features = []
fwd_best_goodness = 0.0

print("\n開始 Forward Feature Selection\n")

while True:
    feature_to_add = None
    best_feature_goodness = 0.0

    for feature_index in range(len(Attributes)):
        if feature_index not in fwd_selected_features:
            # 複製已選Attributes列表，並加入候選Attributes
            features_to_try = fwd_selected_features.copy()
            features_to_try.append(Attributes[feature_index])

            # 提取候選Attributes
            candidate_features = []
            for feature in features_to_try:
                all_feature = []
                for sample in discr_freq_data:
                    select_raw_data = sample[feature]
                    all_feature.append(select_raw_data)
                candidate_features.append(all_feature)

            # 計算Goodness值
            goodness = Goodness(candidate_features, features_to_try)

            # 如果Goodness值更高，則更新候選Attributes和Goodness值
            if goodness > best_feature_goodness:
                best_feature_goodness = goodness
                feature_to_add = feature_index


    # 如果沒有更好的Attributes可加入，則退出迴圈
    if best_feature_goodness < fwd_best_goodness:
            break

    # 增加最佳Attributes到已選Atttibutes中，並更新Goodness值
    fwd_selected_features.append(feature_to_add)
    fwd_best_goodness = best_feature_goodness


    # 輸出已選Attributes subset和Goodness值
    selected_feature_names = [column_names[i +1] for i in fwd_selected_features]
    print(f"Selected Features: {selected_feature_names}")
    print(f"Best Goodness: {fwd_best_goodness:.4f}")

print("\nForward Feature Selection完成。最佳特徵列名:\n", selected_feature_names)


# 初始化已選Attirbutes和Goodness值
bwd_selected_features = list(range(len(Attributes)))
bwd_best_goodness = 0.0

print("\n開始 Backward Feature Selection\n")
while True:
    feature_to_remove = None
    best_feature_goodness = 0.0

    for feature_index in bwd_selected_features:
        # 複製已選Attributes列表，並刪除候選Attributes
        features_to_try = bwd_selected_features.copy()
        features_to_try.remove(feature_index)

        # 提取候選Attributes
        candidate_features = []
        for feature in features_to_try:
            all_feature = []
            for sample in discr_freq_data:
                select_raw_data = sample[feature]
                all_feature.append(select_raw_data)
            candidate_features.append(all_feature)

        # 計算Goodness值
        goodness = Goodness(candidate_features, features_to_try)

        # 如果Goodness值更高，則更新候選Attributes和Goodness值
        if goodness > best_feature_goodness:
            best_feature_goodness = goodness
            feature_to_remove = feature_index
            


    # 如果没有更差的Attributes可以刪除，則退出迴圈
    if bwd_best_goodness >= best_feature_goodness:
        
        break

    # 增加最差的Attributes到已選Attributes中，並更新Goodness值
    bwd_selected_features.remove(feature_to_remove)
    bwd_best_goodness = best_feature_goodness

    selected_feature_names = [column_names[i + 1] for i in bwd_selected_features]
    print(f"Selected Features: {selected_feature_names}")
    print(f"Best Goodness: {bwd_best_goodness:.4f}")


print("\nBackward Feature Selection完成。最佳特徵列名:\n", selected_feature_names)

selected_data = []
for i in range(len(discr_freq_data)):
    selected_ins = {} 
    for j in fwd_selected_features:
        feature_value = discr_freq_data[i][j]
        selected_ins[j] = feature_value
    selected_data.append(selected_ins)


# 计算类别的先验概率
prior_probs = {}  # 存储各个类别的先验概率

for class_label in set(Ground_tru):
    prior_probs[class_label] = (Ground_tru.count(class_label)) #/ len(Ground_tru)

# 训练模型：计算每个属性在每个类别中的条件概率
attribute_probs = {}

# Initialize conditional probabilities to 0 for all attribute values and class labels
for attr in fwd_selected_features:
    attribute_probs[attr] = {}
    # for class_label in set(Ground_tru):
    #     attribute_probs[attr][class_label] = {}
    for value in range(1, 11):  # Initialize for attribute values '1' to '10'
        attribute_probs[attr][str(value)] = 0

# Count the occurrences of each attribute value within each class
# for sample, class_label in zip(discr_freq_data, Ground_tru):
#     for attr in fwd_selected_features:
#         value = sample[attr]
#         attribute_probs[attr][class_label][value] += 1
            
# for attr in fwd_selected_features:
#     attribute_probs[attr] = {val: 0 for val in set(sample[attr] for sample in discr_freq_data)}

# 统计每个类别中每个属性值的计数
for sample, class_label in zip(discr_freq_data, Ground_tru):
    for attr in fwd_selected_features:
        value = sample[attr]
        # if class_label not in attribute_probs[attr]:
        #     attribute_probs[attr][class_label] = 0
        # if value not in attribute_probs[attr]:
        #     attribute_probs[attr][value] = 0
        attribute_probs[attr][value] += 1


# 计算条件概率
for attr in fwd_selected_features:
    for class_label in attribute_probs[attr]:
        for value in attribute_probs[attr][class_label]:
            attribute_probs[attr][class_label][value] = attribute_probs[attr][class_label][value] / (Ground_tru.count(class_label) + len(attribute_probs[attr]))

# 进行分类
def classify(sample):
    class_probabilities = {}  # 存储每个类别的概率

    for class_label in prior_probs:
        prob = math.log(prior_probs[class_label])
        for attr in Attributes:
            value = sample[attr]
            if class_label in attribute_probs[attr] and value in attribute_probs[attr][class_label]:
                prob += math.log(attribute_probs[attr][class_label][value])
        class_probabilities[class_label] = prob

    max_prob_label = max(class_probabilities, key=class_probabilities.get)
    return max_prob_label

# 测试分类
test_sample = raw_data[0]  # 使用第一个样本进行测试
result = classify(test_sample)
print(f"Classification result: {result}")