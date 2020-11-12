##Perion Docker Image Classifier
*Perion Image Classifier as a Docker Container v1*
**Updated: November 12, 2020**

This application is a self contained image classifier model as a docker container. 

### Build and run the Docker Container

You can build and run the docker container on port `:5000` using the following commands

`docker build -t docker-flask:latest .`
`docker run --name <appname> -v$PWD/app:/app -p5000:5000 docker-flask:latest`

### MAIN api interface

`/api # API main call`

Use `/api?url=<data_src>` to define the URL for the dataset. 

> Example: `/api?url=https://example.com/dataset.tgz`


