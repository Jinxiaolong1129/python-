from tqdm import tqdm

text = ""
for char in tqdm(["a", "b", "c", "d"]):
    text = text + char


# pbar = tqdm(["a", "b", "c", "d"])
# for char in pbar:
#     pbar.set_description("Processing %s" % char)