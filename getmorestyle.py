import urllib.request
import os

models = {
    "candy.t7": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/candy.t7",
    "mosaic.t7": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/mosaic.t7",
    "starry_night.t7": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/starry_night.t7",
    "udnie.t7": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/udnie.t7"
}

print("Downloading new styles... this might take a minute.")
for name, url in models.items():
    if not os.path.exists(name):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, name)
    else:
        print(f"{name} is already downloaded!")
print("All done! You are ready to run the main script.")