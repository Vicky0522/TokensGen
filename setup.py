import time
from setuptools import setup, find_packages
import sys
import os
import os.path as osp

WORK_DIR = "longvgen"
NAME = "longvgen"
author = "wenqi.oywq"
author_email = 'wenqi.ouyang@ntu.edu.sg'

version_file = 'longvgen/version.py'

def get_hash():
    if False:#os.path.exists('.git'):
        sha = get_git_hash()[:7]
    # currently ignore this
    # elif os.path.exists(version_file):
    #     try:
    #         from basicsr.version import __version__
    #         sha = __version__.split('+')[-1]
    #     except ImportError:
    #         raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
__gitsha__ = '{}'
version_info = ({})
"""
    sha = get_hash()
    with open('VERSION', 'r') as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = ', '.join([x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split('.')])

    version_file_str = content.format(time.asctime(), SHORT_VERSION, sha, VERSION_INFO)
    with open(version_file, 'w') as f:
        f.write(version_file_str)

REQUIRE = [
]
    
def install_requires(REQUIRE):
    for item in REQUIRE:
        os.system(f'pip install {item}')

write_version_py()
install_requires(REQUIRE)
setup(
    name=NAME,
    packages=find_packages(),
    version=get_version(),
    description="long video generation",
    author=author,
    author_email=author_email,
    keywords=["long video generation"],
    install_requires=[],
    include_package_data=False,
    exclude_package_data={'':['.gitignore','README.md','./configs/','./outputs/'],
                         },
    #package_data = {'VideoRecolor':\
    #                [
    #                 osp.join('data', '*.txt'), \
    #                 osp.join('data', 'filter', '*_lut.txt'), \
    #                 osp.join('data', 'HDR', '*_lut.txt'), \
    #                ]
    #               },
    entry_points={'console_scripts': ['pyinstrument = pyinstrument.__main__:main']},
    zip_safe=False,
)
