import pkg_resources
from os.path import join, basename
from setuptools import find_packages
from cx_Freeze import setup, Executable

def collect_dist_info(packages):
    """
    Recursively collects the path to the packages' dist-info.
    """
    if not isinstance(packages, list):
        packages = [packages]
    dirs = []
    for pkg in packages:
        distrib = pkg_resources.get_distribution(pkg)
        for req in distrib.requires():
            dirs.extend(collect_dist_info(req.key))
        dirs.append((distrib.egg_info, join('Lib', basename(distrib.egg_info))))
    return dirs

setuptools = collect_dist_info("setuptools")
setuptools.append("data")
options = {
    'build_exe': {
        'includes': [
            'cx_Logging', 'idna'
        ],
        'packages': [
            'asyncio', 'flask', 'jinja2', 'dash', 'plotly', 'waitress', 'anndata', 'math', 'matplotlib'
        ],
        'excludes': ['tkinter'],
        "include_files": setuptools
    }
}

executables = [
    Executable('server.py',
               base='console',
               targetName='dash_app_test.exe')
]

setup(
    name='dash_app_test',
    packages=find_packages(),
    version='0.0.1',
    description='rig',
    executables=executables,
    options=options
)