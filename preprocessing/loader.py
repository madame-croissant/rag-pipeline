import re
import os
from tqdm import tqdm

#this handles markdown only
#maybe remove links and add page numbers
#only does for one file

class DataLoader:
    def __init__(self):
        self.dict_list = []

    def build_corpus(self, root_folder):
        """
        process all files
        """

        for root, dirs, files in tqdm(os.walk(root_folder), desc="Files processed"):
            #print("current dir", root)
            #print("subdirs", dirs)
            #print("files", files)
            for f in files:
                if f.endswith('.md'):
                    #print("files", f)
                    full_path = os.path.join(root, f)
                    self.processor(full_path)




    def processor(self, filename):
        """
        Processing each .md file separately, building a dic with source, heading, text
        """

        curr_heading = None
        text = ""

        with open(filename, "r", encoding="utf-8") as f:

            lines = f.readlines()

        lines = self.strip_preamble(lines)

        for line in lines:
    
            heading = re.match(r'^#{1,3}\s+(.+)$', line)

            if heading and curr_heading is None:
            #first heading

                if text.strip():
                    #text before the first heading that is not "---"
                    data = {
                        "source_file": self.path_extractor(filename),
                        "heading": "intro",
                        "text": text
                        }

                    self.dict_list.append(data)
                    text = ""
                    
                curr_heading = heading.group(1)
                continue
                


            if not heading:
                text += line
            else:
                if text.strip():
                    data = dict()
                    path = self.path_extractor(filename)
                    data["source_file"] = path
                    data["heading"] = curr_heading if curr_heading is not None else "Document Root"
                    data["text"] = text

                    self.dict_list.append(data)

                curr_heading = heading.group(1)
                text = ""

        #last block
        if text.strip():
            data = dict()
            path = self.path_extractor(filename)
            data["source_file"] = path
            data["heading"] = curr_heading if curr_heading is not None else "Document Root" 
            data["text"] = text
            
            self.dict_list.append(data)
        
        

    def path_extractor(self, filename):

        path = os.path.basename(filename) 
        return path
    

    def strip_preamble(self, lines):

        if lines and lines[0].strip() == "---":
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    return lines[i+1:] #after closing 
        return lines
        

if __name__ == "__main__":

    file = "data/raw/management.md"
    root_folder = 'data/raw'
    loader = DataLoader()
    #loader.processor(file)
    
    loader.build_corpus(root_folder)
    print("Total chunks:", len(loader.dict_list))

    #print(loader.dict_list[0])
    #print(loader.dict_list[-1])
    #print(loader.dict_list)


