# https://stackoverflow.com/a/51058689
import png
import sys

TEXT_CHUNK_FLAG = b'tEXt'


def generate_chunk_tuple(type_flag, content):
    return tuple([type_flag, content])


def generate_text_chunk_tuple(str_info):
    type_flag = TEXT_CHUNK_FLAG
    return generate_chunk_tuple(type_flag, bytes(str_info, 'utf-8'))


def insert_text_chunk(filein, fileout, text, index=1):
    if index < 0:
        raise Exception('The index value {} less than 0!'.format(index))

    reader = png.Reader(filename=filein)
    chunks = reader.chunks()
    chunk_list = list(chunks)
    print(chunk_list[0])
    print(chunk_list[1])
    print(chunk_list[2])
    chunk_item = generate_text_chunk_tuple(text)
    chunk_list.insert(index, chunk_item)
    print(chunk_list[0])
    print(chunk_list[1])
    print(chunk_list[2])
    with open(fileout, 'wb') as dst_file:
        png.write_chunks(dst_file, chunk_list)


if __name__ == '__main__':
    insert_text_chunk(sys.argv[1],sys.argv[2],  '\n' + open(sys.argv[3]).read(), index=0)
