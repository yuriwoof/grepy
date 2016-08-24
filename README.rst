Grepy
======

This prototype project provides Caffe inference functionality via REST api.

Build
--------

Grepy is provided as Docker container.
So you need to build Docker container at first.

::

  $ cd docker/caffe_cpu
  $ sudo docker build -t local/caffe_cpu -f docker/caffe_cpu/Dockerfile.caffe_cpu .
  $ cd ../grepy_cpu
  $ sudo docker build -t local/grepy_cpu -f Dockerfile.grepy_cpu .

This leads following micro-service.
This managed under supervisord.

::

  [Client]--[Nginx]--[uWSGI]--[server.py(Flask)]--[pretrained CaffeNet(Caffe)]

And run Grepy container.
You can remove the container at stopping continer using ``--rm`` option.

::

  $ sudo docker run --rm -i -t -p 80:80 local/grepy_cpu

Usage
------

You need to prepare test picture. I used `Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_.

::

  $ wget -O - http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz | tar xfz -
  $ curl --form "image=@101_ObjectCategories/airplanes/image_0001.jpg" http://(YOUR_HOST_IP_ADDRESS)/classify


        {
          "result": [
              {
                 "name": "n02690373 airliner",
                 "score": "0.952552318573"
              },
              {
                 "name": "n04552348 warplane, military plane",
                 "score": "0.0273641180247"
              },
              { 
                 "name": "n04008634 projectile, missile",
                 "score": "0.00465240329504"
              }
           ] 
         }


