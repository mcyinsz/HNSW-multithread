from setuptools import setup, Extension
import pybind11
import sys
import os

# 获取当前目录路径
base_dir = os.path.dirname(os.path.abspath(__file__))

# 自动收集源文件
def find_sources(dir, extensions=('.cpp', '.c')):
    sources = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(extensions):
                sources.append(os.path.join(root, file))
    return sources

# 编译配置
ext = Extension(
    'hnswlib._core',
    sources=[
        os.path.join('bind', 'core.cpp'),  # 主绑定文件
        *find_sources('src')               # 自动包含所有源文件
    ],
    include_dirs=[
        pybind11.get_include(),
        os.path.join(base_dir, 'include'),          # 主头文件目录
        os.path.join(base_dir, 'include', 'impl'),   # impl头文件
        os.path.join(base_dir, 'include', 'utils')   # utils头文件
    ],
    language='c++',
    extra_compile_args=[
        '-std=c++17',
        '-O3',
        '-mavx2', '-mfma', '-mavx512f', '-mavx512vl',  # SIMD指令
        '-fopenmp',  # OpenMP支持
        '-Wno-unused-function'  # 抑制警告
    ],
    extra_link_args=['-fopenmp'],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
)

setup(
    name='hnswlib',
    version='0.2',
    packages=['hnswlib'],
    package_dir={'hnswlib': 'bind'},
    ext_modules=[ext],
    install_requires=['pybind11>=2.6', 'numpy'],
    python_requires='>=3.7',
    zip_safe=False
)
