import os
rootdir = '/usr/local/lib/python3.5/dist-packages'
os.chdir(rootdir)

filenames = []

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if '.py' in file and '.pyc' not in file and 'test' not in file:
            filenames.append(os.path.join(subdir, file))

with open('/home/gabriel/fun/char-rnn-tensorflow/data/python/python.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            try:
                outfile.write(infile.read())
            except Exception as e:
                print(e)
                print(fname)


import re

def removeComments(string):
    string = re.sub(re.compile("/\*.*?\*/\n",re.DOTALL ) ,"" ,string) # remove all occurance streamed comments (/*COMMENT */) from string
    string = re.sub(re.compile("#.*?\n" ) ,"" ,string) # remove all occurance singleline comments (#COMMENT\n ) from string
    string = re.sub(re.compile('r"""(.|\n)*?"""\n' ) ,"" ,string) # remove all occurance singleline comments (#COMMENT\n ) from string
    string = re.sub(re.compile('"""(.|\n)*?"""\n' ) ,"" ,string) # remove all occurance singleline comments (#COMMENT\n ) from string
    string = re.sub(re.compile('\n\n' ) ,"\n" ,string) # remove all occurance singleline comments (#COMMENT\n ) from string
    return string

#test_str = "from matplotlib.pylab import *\nimport matplotlib.pylab\n__doc__ = matplotlib.pylab.__doc__\n#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n# Copyright (c) 2005-2010 ActiveState Software Inc.\n# Copyright (c) 2013 Eddy Petrișor\n\n\"\"\"Utilities for determining application-specific dirs.\n\nSee <http://github.com/ActiveState/appdirs> for details and usage.\n\"\"\"\n# Dev Notes:\n# - MSDN on where to store app data files:\n#   http://support.microsoft.com/default.aspx?scid=kb;en-us;310294#XSLTH3194121123120121120120\n# - Mac OS X: http://developer.apple.com/documentation/MacOSX/Conceptual/BPFileSystem/index.html\n# - XDG spec for Un*x: http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html\n\n__version_info__ = (1, 4, 3)\n__version__ = '.'.join(map(str, __version_info__))\n\n\nimport sys\nimport os\n\nPY3 = sys.version_info[0] == 3\n\nif PY3:\n    unicode = str\n\nif sys.platform.startswith('java'):\n    import platform\n    os_name = platform.java_ver()[3][0]\n    if os_name.startswith('Windows'): # \"Windows XP\", \"Windows 7\", etc.\n        system = 'win32'\n    elif os_name.startswith('Mac'): # \"Mac OS X\", etc.\n        system = 'darwin'\n    else: # \"Linux\", \"SunOS\", \"FreeBSD\", etc.\n        # Setting this to \"linux2\" is not ideal, but only Windows or Mac\n        # are actually checked for and the rest of the module expects\n        # *sys.platform* style strings.\n        system = 'linux2'\nelse:\n    system = sys.platform\n\n\n\ndef user_data_dir(appname=None, appauthor=None, version=None, roaming=False):\n    r\"\"\"Return full path to the user-specific data dir for this application.\n\n        \"appname\" is the name of application.\n            If None, just the system directory is returned.\n        \"appauthor\" (only used on Windows) is the name of the\n            appauthor or distributing body for this application. Typically\n            it is the owning company name. This falls back to appname. You may\n            pass False to disable it.\n        \"version\" is an optional version path element to append to the\n            path. You might want to use this if you want multiple versions\n            of your app to be able to run independently. If used, this\n            would typically be \"<major>.<minor>\".\n            Only applied when appname is present.\n        \"roaming\" (boolean, default False) can be set True to use the Windows\n            roaming appdata directory. That means that for users on a Windows\n            network setup for roaming profiles, this user data will be\n            sync'd on login. See\n            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>\n            for a discussion of issues.\n\n    Typical user data directories are:\n        Mac OS X:               ~/Library/Application Support/<AppName>\n        Unix:                   ~/.local/share/<AppName>    # or in $XDG_DATA_HOME, if defined\n        Win XP (not roaming):   C:\\Documents and Settings\\<username>\\Application Data\\<AppAuthor>\\<AppName>\n        Win XP (roaming):       C:\\Documents and Settings\\<username>\\Local Settings\\Application Data\\<AppAuthor>\\<AppName>\n        Win 7  (not roaming):   C:\\Users\\<username>\\AppData\\Local\\<AppAuthor>\\<AppName>\n        Win 7  (roaming):       C:\\Users\\<username>\\AppData\\Roaming\\<AppAuthor>\\<AppName>\n\n    For Unix, we follow the XDG spec and support $XDG_DATA_HOME.\n    That means, by default \"~/.local/share/<AppName>\".\n    \"\"\"\n    if system == \"win32\":\n        if appauthor is None:\n            appauthor = appname\n        const = roaming and \"CSIDL_APPDATA\" or \"CSIDL_LOCAL_APPDATA\"\n        path = os.path.normpath(_get_win_folder(const))\n        if appname:\n            if appauthor is not False:\n                path = os.path.join(path, appauthor, appname)\n            else:\n                path = os.path.join(path, appname)\n    elif system == 'darwin':\n        path = os.path.expanduser('~/Library/Application Support/')\n        if appname:\n            path = os.path.join(path, appname)\n    else:\n        path = os.getenv('XDG_DATA_HOME', os.path.expanduser(\"~/.local/share\"))\n        if appname:\n            path = os.path.join(path, appname)\n    if appname and version:\n        path = os.path.join(path, version)\n    return path\n"
#print(removeComments(test_str))

with open('/home/gabriel/fun/char-rnn-tensorflow/data/python/python.txt', 'r') as myfile:
    data = myfile.read()
    data_cleaned = removeComments(data)
    print(100*len(data_cleaned)/len(data), "%")
    with open('/home/gabriel/fun/char-rnn-tensorflow/data/python/python_cleaned.txt', 'w') as outfile:
        outfile.write(data_cleaned)
