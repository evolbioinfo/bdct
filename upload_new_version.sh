rm -rf bdpn.egg-info build dist
python setup.py sdist bdist_wheel
twine upload dist/* && \
sudo docker build -t evolbioinfo/bdct:v0.1.23 -f Dockerfile . && sudo docker login && sudo docker push evolbioinfo/bdct:v0.1.23