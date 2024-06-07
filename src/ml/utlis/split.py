from typing import List


def get_text_chunks(text: str, splitter):
    chunks = [x for x in splitter.split_text(text)]
    return chunks
