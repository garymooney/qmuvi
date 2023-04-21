@echo off
echo "Generating API documentation with Sphinx..."
poetry run sphinx-apidoc -f -o docs qmuvi/
echo "Building documentation with Sphinx..."
poetry run sphinx-build docs docs/_build
echo "Opening generated HTML documentation in default web browser..."
start "" "docs\_build\index.html"
echo "Done!"

