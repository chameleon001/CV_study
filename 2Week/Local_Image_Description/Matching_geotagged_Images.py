#%%

import json

try: import simplejson as json
except ImportError: import json

#%%
import os
import urllib
from urllib.parse import urlparse
import simplejson as json

#%%

#다운받을수가 없다니.. 사이트가....
url='http://www.panoramio.com/map/get_panoramas.php?order=popularity&\set=public&from=0&to=20&minx=-77.037564&miny=38.896662&\maxx=-77.035564&maxy=38.898662&size=medium'
c = urllib.request.urlopen(url)

print(c)
# get the urls of individual images from JSON
j = json.loads(c.read())
imurls = []

for im in j['photos']:
    imurls.append(im['photo_file_url'])

for url in imurls:
    image = urllib.URLopener()
    image.retrieve(url, os.path.basename(urlparse.urlparse(url).path))
    print ('downloading {}'.format(url))
# %%

