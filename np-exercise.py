import numpy as np
print("=" * 50)
print(f"{" " * 17}First exercise")
print("=" * 50)

gym_data = np.array([ 
    [28, 75, 175, 4], 
    [34, 68, 168, 3], 
    [45, 82, 180, 2], 
    [22, 58, 162, 5], 
    [38, 90, 0, 1], 
    [29, 65, 170, 0] 
])

member_names = np.array(["Ali", "Sara", "Reza", "Neda", "Hassan", "Maryam"])

print("-" * 20)
print("part => 1")
print("-" * 20)

heights = gym_data[:,2]
print(f"heights list is : {heights}")
avg_heights = np.mean(heights[heights != 0])
print(f"avg_heights is: {avg_heights} ")

for i in range(len(gym_data)):
    if gym_data[i,2] == 0:
        gym_data[i,2] = avg_heights
# print(gym_data)


sessions = gym_data[:,3]
avg_sessions = np.mean(sessions[sessions != 0])
print(f"avg_sessions is : {avg_sessions}")
for i in range(len(gym_data)):
    if gym_data[i,3] == 0:
        gym_data[i,3] = avg_sessions
# print(gym_data)

print("-" * 20)
print("part => 2")
print("-" * 20)

new_gym_data = gym_data[:,1:]
print(f"new_gym_data :\n {new_gym_data}")

print("-" * 20)
print("part => 3")
print("-" * 20)

bmi = np.round(new_gym_data[:,0] / ((new_gym_data[:,1] / 100) * (new_gym_data[:,1] / 100)),2)

print(f"bmi list is:\n {bmi}")
fitness_score = ((2 * new_gym_data[:,2]) + (1 * bmi))
print(f"fitness_score list is:\n {fitness_score}") 

print("-" * 20)
print("part => 4")
print("-" * 20)

biggest_fitness_score_name = member_names[np.argmax(fitness_score)]
print(f"biggest_fitness_score_name is: \n{biggest_fitness_score_name}")

print("-" * 20)
print("part => 5")
print("-" * 20)


mean_of_sessions = np.mean(new_gym_data[:,2])
print(f"mean_of_sessions is ==> {mean_of_sessions}")
members_std = np.std(new_gym_data[:,2])
print(f"members_std is ==> {members_std}\n")
z_score = np.abs((new_gym_data[:,2] - mean_of_sessions) / members_std)
print(f"z_score list is:\n{z_score}\n")
print(f"Biggest difference session whith others ==> {member_names[np.argmax(z_score)]}")


print("=" * 50)
print(f"{" " * 17}Second exercise")
print("=" * 50)

recipes = np.array([ 
    [15, 350, 2, 5], 
    [45, 600, 7, 10], 
    [10, 200, 0, 3], 
    [30, 450, 5, 7], 
    [60, 800, 8, 12] 
]) 
recipe_names = ["Salad", "Curry", "Toast", "Pasta", "Stew"] 

print("-" * 20)
print("part => 1")
print("-" * 20)

# mean = np.mean(recipes, axis=0)
max_m = np.max(recipes, axis=0)
min_m = np.min(recipes, axis=0)
scaled_recipes = ((recipes - min_m) / (max_m - min_m))
print(f"scaled_recipes is:\n {scaled_recipes}")

print("-" * 20)
print("part => 2")
print("-" * 20)

users = np.array([ 
    [10, 250, 1, 4], 
    [50, 700, 8, 11], 
    [25, 400, 4, 6] 
]) 

scaled_users = ((users - min_m) / (max_m - min_m))
print(f"scaled_users is:\n{scaled_users}")

print("-" * 20)
print("part => 3")
print("-" * 20)

scaled_recipes_reshape = scaled_recipes.reshape(1,5,4)
scaled_users_reshape = scaled_users.reshape(3,1,4)

diff = scaled_users_reshape - scaled_recipes_reshape
euclidean = np.linalg.norm(diff, axis= 2)
print(f"euclidean distance is:\n {euclidean}")

print("-" * 20)
print("part => 4")
print("-" * 20)

nearest_recipes = np.argmin(euclidean,axis=1)
print(f"nearest_recipes index list is ==> {nearest_recipes}")

for i in range(len(users)):
    print(f"user {i + 1} ==> {recipe_names[nearest_recipes[i]]}")

print("-" * 20)
print("part => 5")
print("-" * 20)

sort_list_recipes = np.argsort(euclidean)
 
print(f"sort_list_recipes:\n {sort_list_recipes}")

for i in range(len(users)):
    print(f"\nuser {i + 1} ==>")
    for j in range(5):
        print(recipe_names[sort_list_recipes[i,j]])

print("=" * 50)
print(f"{" " * 17}Third exercise")
print("=" * 50)

scores = np.array([ 
    [18, 15, 20], 
    [12, 14, 16], 
    [20, 19, 18], 
    [10, 8, 15] 
])

scheme_A = np.array([0.5, 0.3, 0.2]) 
scheme_B = np.array([0.2, 0.3, 0.5]) 
scheme_C = np.array([0.1, 0.2, 0.7])

print("-" * 20)
print("part => 1")
print("-" * 20)

weight_matrix = np.hstack((scheme_A, scheme_B, scheme_C)).reshape(3,3)
print(f"weight_matrix is :\n{weight_matrix}")

print("-" * 20)
print("part => 2")
print("-" * 20)

final_scores = scores @ weight_matrix.T 
print(f"final_scores is :\n{final_scores}")

print("-" * 20)
print("part => 3")
print("-" * 20)

good_scheme = np.argmax(final_scores,axis=1)
print(good_scheme)


print("-" * 20)
print("part => 4")
print("-" * 20)

mean_of_final_scores_of_scheme = np.mean(final_scores,axis=0)
print(mean_of_final_scores_of_scheme)

schemes = ["scheme_A","scheme_B","scheme_C"]
biggest_grade = np.argmax(mean_of_final_scores_of_scheme)
print(f"good scheme is ==> {schemes[biggest_grade]}")

print("-" * 20)
print("part => 5")
print("-" * 20)

x = 0.2
scheme_D = np.array([x, 1-3*x, 2*x])

new_weight_matrix = np.hstack((scheme_A, scheme_B, scheme_C, scheme_D)).reshape(4,3)
print(f"new_weight_matrix is :\n{new_weight_matrix}")



print("=" * 50)
print(f"{" " * 17}Fourth exercise")
print("=" * 50)

weather_data = np.random.randn(2, 8, 4) * 5 + 20 

print("-" * 20)
print("part => 1")
print("-" * 20)

def analyze_day(day):
    print(day)

day2 = weather_data[1].reshape(4,8)
print(f"day2.shape: {day2.shape}")

analyze_day(day2)

print("-" * 20)
print("part => 2")
print("-" * 20)

# weather_data_flattened = weather_data.reshape(-1)
weather_data_flattened = weather_data.flatten()
print(weather_data_flattened.shape)


print("-" * 20)
print("part => 3")
print("-" * 20)


arr = np.array([[1,2],
                [3,4]])

arr_transpose = arr.T
print(f"arr_transpose:\n {arr_transpose} ==> {arr_transpose.shape}")

arr_flatten = arr.flatten()
print(f"arr_flatten is : {arr_flatten} ==> {arr_flatten.shape}")

"""
ترانهاده فقط جای سطر و ستون را تغییر میدهد اما فلت کردن تمام عناصر 
آرایه را پشت سرهم قرار میدهد و انها را تک بعدی میکند.
"""


print("-" * 20)
print("part => 4")
print("-" * 20)

print(f"weather_data shape is ==> {weather_data.shape}")
day3 = np.random.randn(8, 4).reshape(1,8,4)
print(f"day3 shape is ==> {day3.shape}")

weather_data_3day = np.vstack((weather_data,day3))
# weather_data_3day = np.concatenate((weather_data,day3),axis=0)

print(f"weather_data_3day ==> {weather_data_3day.shape}")


print("=" * 50)
print(f"{" " * 17}Fifth exercise")
print("=" * 50)


print("-" * 20)
print("part => 1")
print("-" * 20)

importance = np.array([3, 8, 1, 9, 4, 7])

print(f"importance shape ==> {importance.shape}")

importance1 = importance.reshape(6,1)
print(f"first way ==> {importance1.shape}")
# print(importance1)

importance2 = importance[:,np.newaxis]
print(f"second way ==> {importance2.shape}")
# print(importance2)


print("-" * 20)
print("part => 2")
print("-" * 20)

model_out = np.array([[[0.9]]])

model_out1 = model_out[0,0,0]
print(model_out1)

# یک آرایه نامپای که فقط یک عنصر دارد را به : item() 
# یک اسکالر (عدد معمولی پایتون) تبدیل می‌کند

model_out2 = model_out.item()
print(model_out2)

print("-" * 20)
print("part => 3")
print("-" * 20)

notes = np.array([5, 10, 15, 20, 25]) 
pinned = notes[1:3] 
pinned[0] = 999 
print(notes)    # [5, 999, 15, 20, 25]

"""
اسلایس کپی نیست بلکه نمایشه در نتیجه آرایه اولیه هم تغییر میکنه.
"""
notes = np.array([5, 10, 15, 20, 25]) 
pinned = notes[1:3].copy() 
pinned[0] = 999 
print(notes)    #[5, 10, 15, 20, 25]



print("-" * 20)
print("part => 4")
print("-" * 20)

notes = np.array([5, 10, 15, 20, 25]) 
pinned = notes[1:3] 
pinned[0] = 999 
print(notes)


print("=" * 50)
print(f"{" " * 17}Sixth exercise")
print("=" * 50)

print("-" * 20)
print("part => 1")
print("-" * 20)

bib_numbers = np.array([101, 102, 103, 104, 105, 106]) 
times_5k = np.array([22.3, 25.1, 21.8, 26.4, 23.0, 24.7])

bib_numbers_2 = np.array([107, 108]) 
times_5k_2 = np.array([20.5, 27.9])

total_bib_numbers = np.concatenate((bib_numbers,bib_numbers_2))
print(f"total_bib_numbers:\n {total_bib_numbers}")

total_times_5k = np.concatenate((times_5k,times_5k_2))
print(f"total_times_5k:\n {total_times_5k}")

print("-" * 20)
print("part => 2")
print("-" * 20)

indices = np.random.permutation(len(total_bib_numbers))

shuffled_runners = total_bib_numbers[indices]
shuffled_times = total_times_5k[indices]
print(shuffled_runners)
print(shuffled_times)

print("-" * 20)
print("part => 3")
print("-" * 20)


time_sorted = np.argsort(total_times_5k)

print(f"total_times_5k:\n {total_times_5k}")
print(f"time_sorted : {time_sorted}")

# for i in range(len(total_bib_numbers)):
#     print(f"runner {total_bib_numbers[i]} ==> {total_times_5k[i]}")

for rank,i in enumerate(time_sorted,start=1):
    print(f"rank {rank} => runner {total_bib_numbers[i]} ==> {total_times_5k[i]}")


print("-" * 20)
print("part => 4")
print("-" * 20)

# bib = int(input("Enter bib number: "))
bib = 104

find_index = np.where(total_bib_numbers == bib)[0][0]    #3
time_of_bib = np.where(time_sorted == find_index)[0][0]     #6
print(f"rank runner {bib} is ==> {time_of_bib + 1}")


print("=" * 50)
print(f"{" " * 17}Seventh exercise")
print("=" * 50)

print("-" * 20)
print("part => 1")
print("-" * 20)

X_messages = np.array([ 
    [12, 0, 1], 
    [45, 5, 8], 
    [8, 0, 0], 
    [30, 3, 4] 
]) 
w = np.array([0.1, 0.8, 0.5]) 
b = -2.0

final = X_messages @ w + b
# final = np.dot(X_messages,w) + b

print(f"Raw output for part 1 ==> {final}")

print("-" * 20)
print("part => 2")
print("-" * 20)

# final = X_messages @ w + b
relu_final = np.maximum(0,final)

print(f"Output with RELU activation function ==> {relu_final}")

print("-" * 20)
print("part => 3")
print("-" * 20)

# === first way ===

threshold = 5
for index,i in enumerate(relu_final,start=1):
    if i <= threshold:
        print(f"message {index} ==> Calm Message")
    else:
        print(f"message {index} ==> Energetic message")


# === second way ===

# for index,i in enumerate(relu_final,start=1):
#     if i <= 0 :
#         print(f"message {index} ==> Calm Message")
#     else:
#         print(f"message {index} ==> Energetic message")



print("\n" + "🎈" * 50 + "\n" + "🎈" * 50 )