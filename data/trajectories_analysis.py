import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    file = open("trajectory_length.txt", "r")
    content = file.readlines()
    trajectories_length = {}

    for index in range(len(content)):
        
        if index <= 3:
            continue
        
        word_list = content[index].split(" ")
        trajectory_length = word_list[5]
        trajectories_count = int(word_list[7])

        trajectories_length[trajectory_length] = trajectories_count


    plt.title("Distribution over the trajectory length", fontsize=32)
    plt.figure(figsize=(34, 20))
    plt.xticks(rotation = 0)
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=26)
    plt.bar(list(trajectories_length.keys()), trajectories_length.values(), color='g', label = trajectories_length.values())
    plt.savefig("trajectory_length.pdf")


        



