# This script should become instructions the gitlab-ci.yaml that uses a revision / branch argument (sha1 etc.)
cd ..; git archive --output docker/sf_datalake.tar.gz HEAD; cd docker/
docker build --no-cache --tag pybuild-datalake .
docker run --rm -i pybuild-datalake cat /python_packages.tar.gz > python_packages.tar.gz
# docker rm pybyuild-datalake
