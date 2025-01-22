---
layout: post
title: Creating An application with Terraform, Kubernetes, and Docker
date: 2025-01-21 22:39 -0500
---

I wanted to create an application called [Touch Grass DC](https://touchgrassdc.com/).  A web app where users can discover new things to do in the DC area.

I wanted to create a microservice architecture and use Docker Compose to manage the containers.  I wanted to be able to run the containers locally and then deploy them to a cloud environment.  I wanted to be able to easily manage the containers and the dependencies between them.

I started by creating a Dockerfile for the web app.  I then created a Dockerfile for the API.  I then created a Dockerfile for the database.  I then created a Dockerfile for the Redis cache.  I then created a Dockerfile for the Nginx reverse proxy.

I then created a docker-compose.yml file to manage the containers.  I then created a Terraform configuration to deploy the containers to a cloud environment.  I then created a Kubernetes configuration to deploy the containers to a cloud environment.

## Digital Ocean
I used Digital Ocean to host the application.  I wanted to make this project PAAS (Platform as a Service) so I could easily manage the application.  I make a simple Terraform script to create the kubernetes cluster and load balancer.  I used github for free Docker Image hosting (insteaf of docker hub).
I used github actions to build the docker images and push them to the github container registry.  Then kubernetes would pull the images from the github container registry.

## Ingress and HTTPS
I wanted to use ingress to route traffic to the application.  I wanted to use HTTPS to secure the traffic.  I wanted to use a wildcard certificate from Let's Encrypt.  I didn't want to have to manually renew the certificate.

I created a Kubernetes Ingress resource to route traffic to the application.  I created a Kubernetes Certificate resource to request the wildcard certificate from Let's Encrypt.  I created a Kubernetes Secret resource to store the certificate and key.

**Remember:** You need a ingress controller to use ingress.  I used the [nginx ingress controller](https://kubernetes.github.io/ingress-nginx/).

## Analytics 
I used grafana, prometheus, and kibana to monitor the application.  I used the [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) to get the metrics from the Kubernetes cluster.  I used the [prometheus-operator](https://github.com/prometheus-operator/prometheus-operator) to deploy the prometheus and grafana.  I used fluentd to collect the logs from the containers and send them to loki.

## Filament
I chose [Filament](https://filamentphp.com/) as an admin portal to allow editing of backend data. Filament is great for editing as well as searching your models in a nice pre-build UI.

Source Code: [https://github.com/ajn123/TouchGrassDC](https://github.com/ajn123/TouchGrassDC)
