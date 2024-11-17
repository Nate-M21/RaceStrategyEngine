"""Module to use OpenF1 and transform the data into suitable format for the RaceDataPacket"""

from urllib.request import urlopen
import json

# Using meeting_key with latest parameter
response = urlopen('https://api.openf1.org/v1/stints?meeting_key=1248')
data = json.loads(response.read().decode('utf-8'))
print(data)
print()
for i in data:
    print(data)