#!/bin/bash
echo Did you remember to increment the version number?
python setup.py sdist
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
