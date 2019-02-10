import requests

responce = requests.post("https://xbackend.appspot.com/fallCamera", json={"status":"true"})
# status code is an int
print(responce.status_code)