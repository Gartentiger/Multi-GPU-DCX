import json
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    raise "Bad args"

with open(sys.argv[1], "r", encoding="utf-8") as f:
    daten = json.load(f)

keys = {}
for i,(key,value) in enumerate(daten["data"]["root"].items()):
    if i == 0:
        continue
    keys.update({key:value["statistics"]["max"][0]})
    print(key,value)
print(keys.keys())
print(keys.values())
plt.bar(keys.keys(), keys.values())
plt.xlabel("Test Data")
plt.ylabel("ms")
plt.savefig("plot.png")