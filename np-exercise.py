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
print("part => 4")
print("-" * 20)


























