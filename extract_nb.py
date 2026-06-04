import json
with open('resnet50-yolo-lastfinal.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
code = '\n\n'.join([''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'])
with open('notebook_code.py', 'w', encoding='utf-8') as f:
    f.write(code)
