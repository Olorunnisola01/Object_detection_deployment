# Object detection app
This app enables you to perform object detection by uploading an image. You can modify the confidence threshold and observe the impact it has on the predictions.

<p align="center">
<img src="https://github.com/Olorunnisola01/Object_detection_deployy/blob/main/images/usage.jpg" width="700">
</p>

## Run with Docker
From the root dir:
```
    docker build -t Olorunnisola01/Object_detection_deployment.
    docker run -p 8501:8501 Olorunnisola01/Object_detection_deployment:latest
```
Then visit [localhost:8501](http://localhost:8501/)

## Development
Using [devcontainer](https://code.visualstudio.com/docs/remote/containers), see basic python [example repo](https://github.com/microsoft/vscode-remote-try-python) and [more advanced repo](https://github.com/microsoft/python-sample-tweeterapp). Use [this streamlit docker template](https://github.com/MrTomerLevi/streamlit-docker). Note you cannot docker run at the same time as the devcontainer as the ports clash.

## References
I took a lot of inspiration from [this article](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/) by Adrian Rosebrock.
