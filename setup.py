from setuptools import setup, find_packages

setup(name='orthogonalx_augm',
      version='1.0.0',
      description='Orthogonal-by-construction model augmentation built on the jax-sysid toolbox',
      author='Bendeguz Gyorok',
      author_email='gyorokbende@sztaki.hu',
      packages=find_packages(),
      install_requires=[
            "jax-sysid==1.0.1",
            "seaborn==0.13.2",
            'PyQt5 ; platform_system=="Linux"',
        ]
      )
