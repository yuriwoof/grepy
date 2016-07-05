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

You need to prepare picture.

::

  $ wget -O - http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz | tar xfz -
  $ curl --form "image=@101_caltech/101_ObjectCategories/airplanes/image_0001.jpg" http://10.83.170.
  {
    "resutl": [
      [
        0.952552318572998,
        "n02690373 airliner"
      ],
      [
        0.02736411802470684,
        "n04552348 warplane, military plane"
      ],
      [
        0.004652403295040131,
        "n04008634 projectile, missile"
      ]
    ]
  }
