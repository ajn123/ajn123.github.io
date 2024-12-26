---
title: Working with Docker Compose
tags: [docker, docker compose, microservices]
---



![Breakfast](../assets/images/breakfast.png)


Trying to build a microservice architecture and I want to use Docker Compose to manage the containers.  I want to be able to run the containers locally and then deploy them to a cloud environment.  I want to be able to easily manage the containers and the dependencies between them.

This all started when I started a project to [broadcast events in the DMV.](https://github.com/ajn123/BreakfastClub)

I learned so much about Docker and Docker Compose while working on this project. I wanted to share some of the things I learned.

# 1 Berware of Volume

```yaml

volumes:
- ./data:/data

volumes:
  db_data:
```
Volumes are a way to persist data across container restarts.  They are stored in the host machine and are not deleted when the container is stopped.  This is useful for things like databases and other persistent data.
One huge problem with volumes is that they are not shared between containers.  This means that if you have a database and you want to backup the data you need to do it manually.
Another problem is that they overwrite the data in the volume when the container is restarted.  This means that if you have a database and you want to backup the data you need to do it manually.
It ALSO overwrites the commands you do in the container.  This means if you copy and paste commands from the container into the host machine you will overwrite the data in the volume.


# 2 Make sure you have the correct permissions
I can't express how important this is.  You can get many errors if you don't have the correct permissions.

```bash
sudo usermod -aG docker $USER
newgrp docker
chown -R $USER:$USER .
```

This will give you the correct permissions to the files and directories.

# 3 Make sure you .env file works for all environments

your .env file should work for all environments.  This means that you should be able to run the same command in the same environment and get the same results.
What this means is that if you have a database in Docker it will have a host name and you have to make sure that the .env file has the correct host name.  But if you are running the same command in a local environment you need to make sure that the .env file has the correct host name for the local environment.


# 4 understand multi stage builds

mult stage builds are a way to build a docker image in multiple stages.  This is useful for building a docker image in a way that is easy to understand and maintain.
The important thing to understand is that the first stage is the build stage and the second stage is the run stage.  This means that the build stage is the stage that builds the docker image and the run stage is the stage that runs the docker image.  None of the files from the build stage are available in the run stage UNLESS you copy them over yourself.


# 5 understand the difference between build and up (and --no-cache)
when you do 'docker compose build' it will build the docker image.  But if you do 'docker compose up' it will build the docker image and then run it.  This means that if you make a change to the docker file you need to do 'docker compose build' before you can run the container.  When rerunning build it will use the cached image and not build the docker image from scratch.
If you do 'docker compose up --no-cache' it will build the docker image from scratch and then run it.  This means that if you make a change to the docker file you need to do 'docker compose up --no-cache' before you can run the container.


# 6 understand the COPY command
the command command can look simple but it can be tricky.  If you are copying a file from the host machine to the container you need to make sure that the file exists in the host machine.  If you are copying a file from the build stage to the run stage you need to make sure that the file exists in the build stage.
Also understand the importance of slashes in the path.  If you are copying a file from the host machine to the container you need to make sure that the file exists in the host machine. 

```yaml
COPY --from=build-stage /app/build /app/build
```

This means that the file exists in the build stage and the run stage.  If you are copying a file from the host machine to the container you need to make sure that the file exists in the host machine.  If you are copying a file from the build stage to the run stage you need to make sure that the file exists in the build stage.

```
COPY /api/build /api 
```
results in the file being copied from the host machine to the container.

```
COPY /api/build/ /api 
```
The slash at the end of the path is important.  It means that the file(s) and not the directory is being copied.