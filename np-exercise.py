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

print("-" * 10)
print("part => 1")
print("-" * 10)

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

print("-" * 10)
print("part => 2")
print("-" * 10)

new_gym_data = gym_data[:,1:]
print(f"new_gym_data :\n {new_gym_data}")

print("-" * 10)
print("part => 3")
print("-" * 10)

bmi = np.round(new_gym_data[:,0] / ((new_gym_data[:,1] / 100) * (new_gym_data[:,1] / 100)),2)

print(f"bmi list is:\n {bmi}")
fitness_score = ((2 * new_gym_data[:,2]) + (1 * bmi))
print(f"fitness_score list is:\n {fitness_score}") 

print("-" * 10)
print("part => 4")
print("-" * 10)

biggest_fitness_score_name = member_names[np.argmax(fitness_score)]
print(f"biggest_fitness_score_name is: \n{biggest_fitness_score_name}")

print("-" * 10)
print("part => 5")
print("-" * 10)


mean_of_sessions = np.mean(new_gym_data[:,2])
print(f"mean_of_sessions is ==> {mean_of_sessions}")
members_std = np.std(new_gym_data[:,2])
print(f"members_std is ==> {members_std}\n")
z_score = np.abs((new_gym_data[:,2] - mean_of_sessions) / members_std)
print(f"z_score list is:\n{z_score}\n")
print(f"Biggest difference session whith others ==> {member_names[np.argmax(z_score)]}")














