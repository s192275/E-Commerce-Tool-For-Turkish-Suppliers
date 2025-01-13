## E-Commerce Tool For Turkish Suppliers
This tool is designed for Turkish e-commerce suppliers and is designed to edit product images and descriptions. In this way, it is aimed to provide equal opportunities among suppliers on e-commerce platforms. The Real-ESRGAN algorithm and CLAHE algorithms are aimed to correct distortions in product images, and the LLaMA 3 LLM model is intended to make product descriptions more suitable for the e-commerce ecosystem.

### Installation

First you should create a virtual environment after installing the project. You should create virtual environment because there may be a conflict due to library installations. 

**To create a virtual environment you can use this command sequentially : **
```python
  python -m venv venv
  venv\Scripts\activate
```
After using this command your virtual environment should be ready. If it is not you can use this [link](https://stackoverflow.com/questions/43069780/how-to-create-virtual-env-with-python-3)

## Running the Requirements File
To do this you should use 
```python
  pip pip install -requirements.txt
```
if this doesn't work you should try install all libraries by manually using pip install <lib_name> command

## Running The Project 
You can run the gradio server using 
```python
  python app.py
```
## Creating .env File 
I worked with LLaMA3 model. To use this you have a HuggingFace token. You can get yourself using this [link](https://www.geeksforgeeks.org/how-to-access-huggingface-api-key/) After getting a key you should create a .env file and you should initialize a variable like ** HF_TOKEN = "YOUR_API_KEY" 

## In-App Images
![alt text](https://github.com/s192275/E-Commerce-Tool-For-Turkish-Suppliers/blob/main/taxi_driver.png?raw=true)
![alt text](https://github.com/s192275/E-Commerce-Tool-For-Turkish-Suppliers/blob/main/taxi_driver_aciklama.png?raw=true)
![alt text](https://github.com/s192275/E-Commerce-Tool-For-Turkish-Suppliers/blob/main/recel.png?raw=true)
![alt text](https://github.com/s192275/E-Commerce-Tool-For-Turkish-Suppliers/blob/main/recel_aciklama.png?raw=true)
