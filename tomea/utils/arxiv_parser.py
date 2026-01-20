import arxiv 
import fitz 
import os 
import re 


def get_paper_data(arxiv_id: str):
    print(f"Fetching metadata for ArXiv ID: {arxiv_id}....")
    try:
        #using arxiv client
        client = arxiv.Client()
        search = arxiv.Search(id_list = [arxiv_id])
        paper = next(client.results(search))

        #temp
        filename = f"paper_{arxiv_id}.pdf"
        print(f"Downloading PDF to {filename}....")
        paper.download_pdf(filename=filename)

        #text extraction
        print("Extracting Text From PDF....")
        doc = fitz.open(filename)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        doc.close()
        os.remove(filename)

        cutoff_index = -1

        len_text = len(full_text)

        last_chunk_start = int(len_text * 0.7)
        first_chunk_start = int (len_text * 0.3)
        
        last_chunk = full_text[last_chunk_start:]
        first_chunk = full_text[:first_chunk_start]
        
        #finding the last "references" in the paper  by looking in the last 30%
        matches = list(re.finditer(r'\nReferences\n|\nBibliography\n', last_chunk, re.IGNORECASE))
        matches_first = list(re.finditer(r'\nAbstract\n', first_chunk, re.IGNORECASE))

        if matches:
            relative_pos = matches[-1].start()
            cutoff_index = last_chunk_start + relative_pos
            print("Truncated 'References' Section")
            cleaned_text = full_text[:cutoff_index]
        else:
            cleaned_text = full_text

        if matches_first:
            relative_pos = matches_first[0].start()
            cutoff_index_start = relative_pos
            print("Truncated Pre-Abstract Section")
            cleaned_text = cleaned_text[cutoff_index_start:]
        else:
            cleaned_text = cleaned_text

        return {
            "title": paper.title,
            "arxiv_id": arxiv_id,
            "url": paper.pdf_url,
            "text": cleaned_text
        }
    except Exception as e:
        print(f"❌ Error processing ArXiv ID {arxiv_id}: {e}")
        return None
    

if __name__ == "__main__":
    data = get_paper_data("1706.03762")
    if data:
        print(f"\n Succesfully loaded Paper: ")
        print(f"Title: {data['title']}")
        print(f"Text Length: {len(data['text'])}")
        print(f"Sample, first 500 characters:\n{(data['text'])[:500]}...... ")




