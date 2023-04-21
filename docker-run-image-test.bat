REM Run the Docker container, mounting the package directory as a volume
docker run -it -v %cd%:/home/runner/work/qmuvi/qmuvi qmuvi-image /bin/bash -c "tox -e py310 ; /bin/bash"