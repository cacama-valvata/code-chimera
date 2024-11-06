# coding=utf-8
'''
该文件演示了如何使用 pyyaml 读写 yaml
'''
import unittest
from io import StringIO

import yaml as pyyaml # pip install PyYAML

def str_presenter(dumper, data):
    '''
    为了实现 value 是多行的时候能显示为 ｜ 和多行，方便阅读
    from https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
    '''
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


pyyaml.add_representer(str, str_presenter)
# to use with safe_dump:
pyyaml.representer.SafeRepresenter.add_representer(str, str_presenter)

def pyyaml_yaml(instream: object, outstream: object):
    config = pyyaml.safe_load(instream)
    # config type is dict
    pyyaml.safe_dump(config, outstream, encoding="utf-8", allow_unicode=True, sort_keys=True, indent=2)



class NoMergeLoader(pyyaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        return pyyaml.BaseLoader.construct_mapping(self, node, deep=deep)

NoMergeLoader.add_constructor('tag:yaml.org,2002:merge', NoMergeLoader.construct_scalar)

'''
使用如下
        with codecs.open(file_path, 'r', encoding='utf-8-sig') as yaml_file:
            loader = NoMergeLoader(yaml_file)
            try:
                return loader.get_single_data()
            finally:
                loader.dispose()
'''

class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        content = '''
root:
  keyc: |
    multi line1
    multi line2
  keyb: value2
  keya: value1
  keyd: 一个中文
        '''
        content = content.strip()
        self.content = content

        expect = '''
root:
  keya: value1
  keyb: value2
  keyc: |
    multi line1
    multi line2
  keyd: 一个中文
'''
        self.expect = expect

    def test_pyyaml_write_as_expect_read(self):
        instream = StringIO(self.content)
        outstream = StringIO()
        pyyaml_yaml(instream, outstream)
        # outstream.seek(0)
        out = outstream.getvalue()

        expect = self.expect
        expect = expect.lstrip()
        self.assertEqual(expect, out)


if __name__ == '__main__':
    unittest.main()
