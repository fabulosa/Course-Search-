import pandas as pd
import pickle
import json


if __name__ == '__main__':
    data = pd.read_csv('course_description_final.tsv', header=0, sep='\t')
    data = data.loc[data['course_description'].notnull(), ['course_subject', 'course_num', 'course_description', 'course_title']]
    data['len'] = data['course_description'].str.split(' ').apply(lambda x: len(x))
    data = data.loc[data['len']>5]
    data['subject_num'] = data['course_subject'] + ' ' + data['course_num']
    data['title_des'] = data['course_title'] + ': ' + data['course_description']
    data = data.loc[:, ['subject_num', 'title_des']].drop_duplicates()
    course_id = {}
    for i in data['subject_num'].tolist():
        if i not in course_id:
            course_id[i] = len(course_id)
    id_course = dict(zip(course_id.values(), course_id.keys()))
    courseId_des = {}
    for i in id_course.keys():
        courseId_des[i] = data.loc[data['subject_num']==id_course[i]]['title_des'].tolist()[0]
    with open('course_id.pkl', 'wb') as f:
        pickle.dump({'course_id': course_id, 'id_course': id_course}, f)
    with open('courseId_description.json', 'w') as f:
        json.dump(courseId_des, f)