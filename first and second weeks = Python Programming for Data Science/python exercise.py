
#################################################
##############   PYTHON EXERCISE   ##############
#################################################


### TASK1
# Examine the data structures of the given values. Use the Type() method.

x = 8
y = 3.2
z = 8j + 18
a = "Hello World"
b = True
c = 23 < 22
l = [1,2,3,4]
d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
t = ("Machine Learning", "Data Science")
s = {"Python", "Machine Learning", "Data Science"}


### TASK 2

# Convert all letters of the given string to uppercase. Replace commas and dots with spaces. Split the text into words.
# Expected output: Use string methods.

text = "The goal is to turn data into information, and information into insight."
text.upper().replace(",", " ").replace(".", " ").split()



### TASK 3


# Apply the following steps to the given list:

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]


# Step 1: Check the number of elements in the list.

len(lst)

# Step 2: Call the elements at index 0 and index 10.

lst[0]
lst[10]

# Step 3: Create a new list from the given list containing ["D", "A", "T", "A"].

data_list = lst[0:4]

# Step 4: Remove the element at index 8.

lst.pop(8)

# Step 5: Add a new element.

lst.append("OFA")

# Step 6: Repeat the element "N" at index 8.

lst.insert(8, "N")

### TASK 4

# Apply the following steps to the given dictionary:

dict = {"Christian":["America", 18],
        "Daisy":["England", 12],
        "Antonio":["Spain", 22],
        "Dante":["Italy", 25]}

# Step 1: Access key values.

dict.keys()

# Step 2: Access values.

dict.values()

# Step 3: Update the value of "Daisy" key from 12 to 13.

dict["Daisy"][1] = 13

# Step 4: Add a new key-value pair with the key "Ahmet" and value ["Turkey", 24].

dict.update({"Ahmet": ["Turkey", 24]})

# Step 5: Remove "Antonio" from the dictionary.

dict.pop("Antonio")

### TASK 5

# Given a list, separate even and odd numbers into two lists.

l = [2, 13, 18, 93, 22]

def func(list):
    odd_list = []
    even_list = []

    for i in list:
        if i %2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list, odd_list

even_list, odd_list  = func(l)


### TASK 6

# In the given list, the names of students who succeeded in the engineering and medical faculties are listed.
# The first three students represent the success order of the engineering faculty, and the last three students represent the order of medical faculty.
# Using enumerate, print the student ranks by faculty. Expected output:.

students = ["Ali", "Veli", "Ayse", "Talat", "Zeynep", "Ece"]

for i, student in enumerate(students):
    if i <= 2:
        faculty = "Engineering"
    else:
        faculty = "Medicine"

    print(f"{faculty} Faculty {i % 3 + 1}. Order: {student}")

### TASK 7
# Print the credit and quota information of courses with the given course codes.

course_code = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
credit = [3, 4, 2, 4]
capacity = [30, 75, 150, 25]

for course, credit, capacity in zip(course_code, credit, capacity):
    print(f"Credit for the course {course}: {credit}, Quota: {capacity}")


### TASK 8
# Given two sets, if the first set contains the second set, print the common elements;
# otherwise, print the difference between the second set and the first set.
# Expected Output: Use the issuperset() method to check inclusion, and use the intersection and difference methods for different and common elements.

ex_set1 = set(["data", "python"])
ex_set2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

def set_disc(set1, set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

set_disc(ex_set1, ex_set2)