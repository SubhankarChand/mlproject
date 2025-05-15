from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT='-e .'
def get_requirements(file_path: str) -> List[str]:

    """
    This function will return the list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace ("\n","")for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements

# This is the setup configuration for the ML project
setup(
    name='mlproject',
    version='0.0.1',
    author='subhankar',
    author_email='subhankarchand66@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)