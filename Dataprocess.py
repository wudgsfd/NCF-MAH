import pandas as pd

appendix_data = pd.read_csv(
    "../../../../../../新建文件夹 (2)/WeChat Files/wxid_ypt32a9iri7f22/FileStorage/File/2024-04/appendix.csv")
appendix_data = appendix_data[["Participants (Course Content Accessed)", "Course Number"]]

user_dict = {}
for i in appendix_data["Participants (Course Content Accessed)"]:
    if i not in user_dict:
        user_dict[i] = len(user_dict)
course_dict = {}
for j in appendix_data["Course Number"]:
    if j not in course_dict:
        course_dict[j] = len(course_dict)

user_dict = {}
course_dict = {}

new_data = {}
new_user = []
new_course = []
for i in appendix_data["Participants (Course Content Accessed)"]:
    new_user.append(user_dict[i])
for i in appendix_data["Course Number"]:
    new_course.append(course_dict[i])

new_data["users"] = new_user
new_data["course"] = new_course
df = pd.DataFrame(new_data)
df.to_csv("train_data.csv")
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size=0.15, random_state=42)
X_train.to_csv("train.csv", index=False)
X_test.to_csv("test.csv", index=False)

import pandas as pd
import random
from tqdm import tqdm

# 读取数据
data = pd.read_csv('test.csv')

# 构建负样本
negative_samples = []
for user in tqdm(data['users'].values, desc="Generating Negative Samples"):
    courses_for_user = data[data['users'] == user]['course'].tolist()
    other_courses = set(data['course'].unique()) - set(courses_for_user)
    temp_data = []
    for _ in range(100):
        negative_course = random.choice(list(other_courses))
        temp_data.append(negative_course)
    negative_samples.append({"postitive": (user, courses_for_user[0]), "negative": temp_data})

with open("my.test.negative", "w") as f:
    for i in negative_samples:
        f.write(str(i["postitive"]))
        f.write("\t")
        for index, j in enumerate(i["negative"]):
            f.write(str(j))
            if index != len(i["negative"]) - 1:
                f.write("\t")
        f.write("\n")

train = pd.read_csv("train.csv")

with open("my.train.rating", "w") as f:
    for index, row in train.iterrows():
        f.write(str(row["users"]))
        f.write("\t")
        f.write(str(row["course"]))
        f.write("\t")
        f.write("1")
        f.write("\n")

with open("my.train.rating", "w") as f:
    for index, row in train.iterrows():
        f.write(str(row["users"]))
        f.write("\t")
        f.write(str(row["course"]))
        f.write("\t")
        f.write("1")
        f.write("\n")




