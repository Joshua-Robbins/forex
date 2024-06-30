# forex

## build the image:

```
cd docker
docker build -f Dockerfile.gpu . -t pynet:gpu-latest
cd ..
```

## use the image:
```
cd models
docker run -v $(pwd):/rundir -it pynet:gpu-latest
$> cd /rundir
#> python train.py
```
