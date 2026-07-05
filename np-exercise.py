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

