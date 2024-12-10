# https://stackoverflow.com/a/51058689
import png
import sys

TEXT_CHUNK_FLAG = b'tEXt'

generate_text_chunk_tuple = lambda str_info: (TEXT_CHUNK_FLAG, bytes(str_info, 'utf-8'))

def insert_text_chunk(filein, fileout, text, index):
    reader = png.Reader(filename=filein)

    chunks = list(reader.chunks())
    my_chunk = generate_text_chunk_tuple(text)
    print([c[0] for c in chunks])
    chunks.insert(index, my_chunk)
    print([c[0] for c in chunks])

    with open(fileout,'wb') as ff:
        png.write_chunks(ff, chunks)

if __name__ == '__main__':
    insert_text_chunk(sys.argv[1],sys.argv[2],  '\n' + open(sys.argv[3]).read(), 1)

