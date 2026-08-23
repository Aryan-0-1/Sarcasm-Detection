import os
import itertools
import xml.etree.ElementTree as ET

def create_csv(file_type):
    inputdir = "../raw_data/" + file_type
    csv = open("../processed_data/OriginalData.csv", "a")

    def parse_file(file_path: str):
        assert type(file_path) is str
        # get title, first index of _
        file_key = '"' + file_path[0:file_path.rfind('_')] + '"'
        file_path = os.path.join(inputdir, file_path)

        # --------- CONVERT AMPERSANDS -------------
        with open(file_path, 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace(' & ', ' ampersand! ')
        filedata = filedata.replace('&', ' ampersand! ')
        filedata = filedata.replace('\n', ' ')


        # Write the file out again
        with open(file_path, 'w') as file:
            file.write(filedata)
        # ------------------------------------------

        csv_points = [file_key]
        with open(file_path, "r") as file:
            it = itertools.chain('<root>', file, '</root>')
            root = ET.fromstringlist(it)
            title = ''
            review = ''
            for index, item in enumerate([(root.find('.//STARS')), (root.find('.//TITLE')), (root.find('.//DATE')), (root.find('.//AUTHOR')),
                         (root.find('.//PRODUCT')), (root.find('.//REVIEW'))]):
                text = item.text
                text = text.replace('"', "'").strip('\n')
                text = text.replace('ampersand!', "&")
                text = text.replace('quot;', "'")
                if index == 1:
                    title = text
                elif index == 5:
                    review = text
                csv_points.append('"' + text + '"')
        title_and_review = '"' + title + '. ' + review + '"'
        csv_points.insert(0, title_and_review)
        return ','.join(csv_points)

    for file_name in os.listdir(inputdir):
        if ".html" in file_name:
            continue
        print('FILE NAME ' + file_name + ' -----------------------------')
        irony_label = '"1"' if file_type == "Ironic" else '"0"'
        csv.write(irony_label + ',' + parse_file(file_name) + '\n')

if __name__ == "__main__":
    open("../processed_data/OriginalData.csv", 'w').close()
    csv = open("../processed_data/OriginalData.csv", "a")
    csv.write("sarcasm_label,text_data,key,stars,title,date,author,product,review\n")
    csv.close()
    create_csv("Regular")
    create_csv("Ironic")
