import os
import shutil
import configparser

cfgfile = r'cfg.ini'
config = configparser.ConfigParser()
config.read(cfgfile)

# Database path from configuration file
db_path = r'%s' % config.get('DB','db_path')


# Class names from configuration file
class_names = [x.strip() for x in (config.get('CLASSES', 'class_names')).split(',')]


test_path = db_path + r'\Test'
training_path = db_path + r'\Training'

# Creating fruit types directories
for name in class_names:
    path1 = test_path + r'\%s' %name
    path2 = training_path + r'\%s' %name
    try:
        if not os.path.exists(path1):
            os.mkdir(path1)
        else:
            os.rename(path1, path1 + " 1")
            os.mkdir(path1)

        if not os.path.exists(path2):
            os.mkdir(path2)
        else:
            os.rename(path2, path2 + " 1")
            os.mkdir(path2)

    except OSError:
        print("Creation of the directory %s failed" % name)
    else:
        print("Successfully created the directory %s " % name)

# Move existing directories
for variety in os.listdir(training_path):

    if variety not in class_names and os.path.isdir(training_path + r'\%s' % variety):
        try:
            fruit_type = variety.split()[0]
            variety_path = training_path + r'\%s' % variety
            type_path = training_path + r'\%s' % fruit_type
            shutil.move(variety_path, type_path)
        except OSError:
            print("Moving directory %s failed" % variety)
        else:
            print("Successfully moved the directory %s " % variety)

for variety in os.listdir(test_path):
    if variety not in class_names and os.path.isdir(test_path + r'\%s' % variety):
        try:
            fruit_type = variety.split()[0]
            variety_path = test_path + r'\%s' % variety
            type_path = test_path + r'\%s' % fruit_type
            shutil.move(variety_path, type_path)
        except OSError:
            print("Moving directory %s failed" % variety)
        else:
            print("Successfully moved the directory %s " % variety)

# Removing unnecessary fruit types
for fruit_type in os.listdir(training_path):
    if fruit_type not in class_names:
        shutil.rmtree(training_path + r'\%s' % fruit_type, ignore_errors=True)

for fruit_type in os.listdir(test_path):
    if fruit_type not in class_names:
        shutil.rmtree(test_path + r'\%s' % fruit_type, ignore_errors=True)