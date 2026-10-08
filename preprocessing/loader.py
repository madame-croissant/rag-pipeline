"""
Data loading and document structure parsing module.

Extracts structured sections from raw Markdown (.md) documents by walking directory
trees, stripping YAML front-matter headers, and parsing content into heading-delimited blocks.
"""


import re
import os
from tqdm import tqdm


class DataLoader:
    """
    Recursively processes and parses Markdown files into structured document dictionaries.

    Attributes:
        dict_list (List[dict[str, str]]): List of extracted document sections containing
            metadata (source_file, heading) and text content.
    """
    
    def __init__(self) -> None:
        """
        Initialises an empty storage list for processed document sections.
        """
        self.dict_list = []

    def build_corpus(self, root_folder: str) -> list[dict[str, str]]:
        """
        Recursively traverses a folder directory and processes all Markdown (.md) files.

        Args:
            root_folder (str): Path to the root directory containing source files.

        Returns:
            List[dict[str, str]]: Populated list of section dictionaries across all files.
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



    def processor(self, filename: str) -> None:
        """
        Parses each Markdown file separately into logical sections.
        Captures introductory text before the first heading under an 'intro' label,
        and groups subsequent text by headings.

        Appends extracted structured records to `self.dict_list`.

        Args:
            filename (str): Full path to the Markdown target file.
        
        """

        curr_heading = None
        text = ""

        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        lines = self.strip_preamble(lines)

        for line in lines:
            heading = re.match(r'^#{1,3}\s+(.+)$', line)

            #First heading encountered, remove the premable 
            if heading and curr_heading is None:

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
                
            # Standard line and heading boundary handling
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

        #Last block handling
        if text.strip():
            data = dict()
            path = self.path_extractor(filename)
            data["source_file"] = path
            data["heading"] = curr_heading if curr_heading is not None else "Document Root" 
            data["text"] = text
            
            self.dict_list.append(data)
        
        

    def path_extractor(self, filename: str) -> str:
        """
        Extracts the base filename from a full file path.

        Args:
            filename (str): Path to the file.

        Returns:
            str: File name string.
        """

        path = os.path.basename(filename) 
        return path
    

    def strip_preamble(self, lines: str) -> list[str]:
        """
        Removes Markdown front-matter headers delimited by '---'.

        Args:
            lines (List[str]): Raw lines from the source file.

        Returns:
            List[str]: Content lines following the closing '---' delimiter,
            or original lines if no front-matter header exists.
        """

        if lines and lines[0].strip() == "---":
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    return lines[i+1:] #after closing 
        return lines
        
