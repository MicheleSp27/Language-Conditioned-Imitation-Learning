import matplotlib.pyplot as plt

def plot_and_txt_file(dictionary,filename):
    file = open(filename + ".txt", "w")
    for key in dictionary.keys():
        file.write("Number of episodes involving " + key + ": " + str(dictionary[key]) + "\n")
    file.close()
    plt.figure(figsize=(24, 20))
    plt.xticks(rotation = 90)
    if filename == "task_counter":
        plt.title("Distribution over the skills", fontsize=32)
        plt.xticks(fontsize=32)
    elif filename == "obj_counter":
        plt.title("Distribution over the objects", fontsize=32)
        plt.xticks(fontsize=13)
    else:
        plt.title("Distribution over the skills and objects", fontsize=32)
    plt.yticks(fontsize=26)
    plt.bar(list(dictionary.keys()), dictionary.values(), color='g', label = dictionary.values())
    plt.savefig(filename + ".pdf")

if __name__ == "__main__":

    file = open("instructions.txt", "r")
    task_counter = {}
    obj_counter = {}
    task_obj_counter = {}

    instructions = file.readlines()
    print("The total number of instructions is: ", len(instructions))

    for index in range(len(instructions)):

        instruction = instructions[index]

        word_list = instruction.split(" ")
        task = word_list[0]
        object = word_list[1].split('\n')[0]
        if len(word_list) > 2:
            next_word = word_list[2]
            next_word = next_word.split('\n')[0]
            if next_word not in ["from", "near", "into", "over","upright", "bottom", "rom"]:
                object = object + " " + next_word
                if len(word_list) > 3:
                    next_word = word_list[3]
                    next_word = next_word.split('\n')[0]
                    if next_word not in ["from", "near", "into", "over", "upright", "bottom", "rom"]:
                        object = object + " " + next_word
                        if len(word_list) > 4:
                            next_word = word_list[4]
                            next_word = next_word.split('\n')[0]
                            if next_word not in ["from", "near", "into", "over", "upright", "bottom", "rom"]:
                                object = object + " " + next_word
        task_object = task + " " + object

        if task_counter.get(task,0) == 0:
            task_counter[task] = 1
        else :
            task_counter[task] = task_counter[task] + 1

        if obj_counter.get(object,0) == 0:
            obj_counter[object] = 1
        else:
            obj_counter[object] = obj_counter[object] + 1
        
        if task_obj_counter.get(task_object,0) == 0:
            task_obj_counter[task_object] = 1
        else:
            task_obj_counter[task_object] = task_obj_counter[task_object] + 1

    plot_and_txt_file(task_counter, "task_counter")
    plot_and_txt_file(obj_counter, "obj_counter")
    plot_and_txt_file(task_obj_counter, "task_obj_counter")
