from PIL import Image
import os
import configparser
cfgfile = r'cfg.ini'
config = configparser.ConfigParser()
config.read(cfgfile)

# Database path from configuration file
db_path = r'%s' % config.get('DB','db_path')


new_size = (32, 32)

for root, dirs, files in os.walk(db_path):
    for file in files:
        if file.endswith((".jpg", ".jpeg")):
            img = Image.open(os.path.join(root, file))
            img.thumbnail(new_size)
            img.convert('RGB').save(os.path.join(root, file))

