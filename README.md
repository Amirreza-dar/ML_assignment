## ML_assignment
An **image classification** project that support 4 different APIs implemented in **Django** web framework.
#### Brief Manual
* in order to run the server you will need to run:

`$ python manage.py runserver`
* API body functions are at `api/views.py` file
* Conncetions of database is at `settings.py` file
* Table creation query can be found at `api/models.py`
* URLs are defined at `api/urls.py`
* Install prerequisites of this project by running `requirements.sh` in your terminal.
* This project is based on local server. site for is on localhost. In order to run see see the interface go to `localhost:8000` or `127.0.0.1:8000/api`.
* default port is set to `8000`. However it might be occupied by other applications. In such case you will need to check terminal log to find the port. 
* In order to test the project it is better to use some `API` testing softwares. [Postman](https://www.postman.com/downloads/) provides a friendly environment to send requests to a given **URL**.
1. You should connect to url (localhost)
2. Use send a link from a **google drive** as header of `GET`. (this application is designed in a way that can only download images from google drive)
3. Use different APIs to communicate with server


* ***NOTE***: You may find new some **test links** in `api/test/test_image_links.txt`
