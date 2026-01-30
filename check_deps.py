#!/usr/bin/env python
"""Check build dependencies"""
import sys

deps = {
    'Cython': False,
    'nanobind': False,
    'scikit_build_core': False,
    'numpy': False,
    'setuptools': False,
    'wheel': False,
}

for name in deps.keys():
    try:
        mod = __import__(name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"[OK] {name}: {version}")
        deps[name] = True
    except ImportError:
        print(f"[NO] {name}: not installed")

print("\n" + "="*50)
print("CYTHON BUILD READY:", deps['Cython'] and deps['numpy'] and deps['setuptools'])
print("NANOBIND BUILD READY:", deps['nanobind'] and deps['scikit_build_core'] and deps['numpy'])
