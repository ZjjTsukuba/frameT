import json, os

man = json.load(open('/home/ghostqw/research/data/PAF/preprocessed/manifest.json'))
files = set(it['file'] for it in man['items'])

missing = []
total = 0
for line in open('frameT/PAF_train.txt'):
    line = line.strip().replace('\\', '/')
    if not line:
        continue
    total += 1
    b = os.path.basename(line)
    if b not in files:
        missing.append(line)

print('total train.txt lines:', total)
print('missing count:', len(missing))
for m in missing:
    print('MISSING:', repr(m))
