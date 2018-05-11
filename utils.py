'''
Created on 5 Mar 2017

@author: af
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.mlab as mlab
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from matplotlib.patches import Polygon as MplPolygon
import seaborn as sns
sns.set(style="white")
from scipy.spatial import ConvexHull
from scipy import stats
from mpl_toolkits.basemap import Basemap, cm, maskoceans
from scipy.interpolate import griddata as gd
import numpy as np
#from matplotlib.patches import Polygon
import pdb
import json
import logging
import pickle
from collections import Counter, OrderedDict
import shapefile
from shapely.geometry import MultiPoint, Point, Polygon, asShape, shape
from collections import defaultdict
import shapely
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


short_state_names = {
       # 'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
       # 'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        #'GU': 'Guam',
       # 'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
      #  'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        #'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

stop_words = ['the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your', 'all', 'have', 'new', 'more', 'an', 'was', 'we', 'will', 'home', 'can', 'us', 'about', 'if', 'page', 'my', 'has', 'search', 'free', 'but', 'our', 'one', 'other', 'do', 'no', 'information', 'time', 'they', 'site', 'he', 'up', 'may', 'what', 'which', 'their', 'news', 'out', 'use', 'any', 'there', 'see', 'only', 'so', 'his', 'when', 'contact', 'here', 'business', 'who', 'web', 'also', 'now', 'help', 'get', 'pm', 'view', 'online', 'c', 'e', 'first', 'am', 'been', 'would', 'how', 'were', 'me', 's', 'services', 'some', 'these', 'click', 'its', 'like', 'service', 'x', 'than', 'find', 'price', 'date', 'back', 'top', 'people', 'had', 'list', 'name', 'just', 'over', 'state', 'year', 'day', 'into', 'email', 'two', 'health', 'n', 'world', 're', 'next', 'used', 'go', 'b', 'work', 'last', 'most', 'products', 'music', 'buy', 'data', 'make', 'them', 'should', 'product', 'system', 'post', 'her', 'city', 't', 'add', 'policy', 'number', 'such', 'please', 'available', 'copyright', 'support', 'message', 'after', 'best', 'software', 'then', 'jan', 'good', 'video', 'well', 'd', 'where', 'info', 'rights', 'public', 'books', 'high', 'school', 'through', 'm', 'each', 'links', 'she', 'review', 'years', 'order', 'very', 'privacy', 'book', 'items', 'company', 'r', 'read', 'group', 'sex', 'need', 'many', 'user', 'said', 'de', 'does', 'set', 'under', 'general', 'research', 'university', 'january', 'mail', 'full', 'map', 'reviews', 'program', 'life']


dialect_state = {
                 
        'atlantic':["Connecticut", "Delaware", "Florida", "Georgia", "Maine", "Maryland", "Massachusetts", "New Hampshire", "New Jersey", "New York", "North Carolina", "Pennsylvania", "Rhode Island", "South Carolina", "Vermont", "Virginia", "Washington, DC"],
        'central':["Arkansas", "Kansas", "Missouri", "Nebraska", "Oklahoma"],
        'central atlantic':["Delaware", "Washington, DC"],
        'delmarva':["Delaware"],
        'desert southwest':["Arizona", "New Mexico"],
        'great lakes':['michigan', 'minnesota', 'wisconsin'],
        'golf states':['alabama', 'florida', 'louisiana', 'mississippi'],
        'inland north':['michigan', 'montana', 'new york', 'washington', 'minnesota', 'north dakota', 'oregon', 'wisconsin'],
        'inland south':["Alabama", "Kentucky", "Mississippi", "Tennessee"],
        'lower mississippi valley':['arkansas', 'mississippi', 'louisiana'],
        'middle atlantic':['maryland', 'south carolina', 'washington, dc', 'north carolina', 'virginia'],
        'midland':['kentucky', 'nebraska', 'tennessee'],
        'mississippi valley':["Arkansas", "Illinois", "Iowa", "Louisiana", "Minnesota", "Mississippi", "Missouri", "Wisconsin"],
        'mississippi-ohio valley':["Illinois", "Indiana", "Iowa", "Kentucky", "Minnesota", "Missouri", "Ohio", "Wisconsin"],
        'new england':["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"],    
        'north':["Connecticut", "Maine", "Massachusetts", "Michigan", "Minnesota", "Montana", "New Hampshire", "New York", "North Dakota", "Oregon", "Rhode Island", "Vermont", "Washington", "Wisconsin"],
        'north atlantic':["Connecticut", "Maine", "Massachusetts", "New Hampshir", "Rhode Island", "Vermont"],
        'north central':["Illinois", "Indiana", "Kentucky", "Michigan", "Ohio", "Wisconsin"],
        'north midland':['nebraska'],
        'northeast':["Connecticut", "Maine", "Massachusetts", "New Hampshire", "New Jersey", "New York", "Rhode Island", "Vermont"],
        'northwest':["Idaho", "Oregon", "Washington", "Montana", "Wyoming"],
        'ohio valley':['kentucky'],
        'pacific':['california', 'washington', 'oregon'],
        'pacific northwest':['washington', 'oregon'],
        'plains states':['nebraska', 'kansas'],
        'rocky mountains':['montana', 'utah', 'idaho', 'nevada', 'wyoming'],
        'south':['florida', 'washington, dc', 'alabama', 'georgia', 'louisiana', 'mississippi', 'north carolina', 'south carolina'],
        'south atlantic':['florida', 'georgia', 'north carolina', 'south carolina'],
        'south midland':['kentucky', 'arkansas', 'tennessee', 'washington, dc', 'west virginia'],
        'southeast':['alabama', 'georgia', 'north carolina', 'tennessee', 'north carolina', 'mississippi', 'florida'],
        'southwest':['arizona', 'new mexico', 'texas', 'oklohama'],
        'upper midwest':['iowa', 'nebraska', 'south dakota', 'north dakota', 'minnesota'],
        'upper mississippi valley':['iowa', 'minnesota', 'wisconsin', 'illinois', 'missouri'],
        'west':["Arizona", "California", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Oregon", "Utah", "Washington", "Wyoming"],
        'west midland':['iowa', 'ohio', 'arkansas', 'tennessee', 'west virginia', 'illinois', 'indiana', 'kentucky', 'nebraska', ]     
                 
                 }

def get_us_city_name():
    #we might exclude words in city names
    all_us_city_names = set()
    with open('~/datasets/shapefiles/us_cities.txt', 'r') as fin:
        for line in fin:
            words = set(line.strip().lower().split())
            for word in words:
                all_us_city_names.add(word)
    return all_us_city_names

def retrieve_location_from_coordinates():

    points = []
    #read points from a file
    with open('./data/latlon_world.txt', 'r') as fin:
        for line in fin:
            line = line.strip()
            lat, lon = line.split('\t')
            lat, lon = float(lat), float(lon)
            point = (lat, lon)
            points.append(point)
    #read point city-countries from http://people.eng.unimelb.edu.au/tbaldwin/resources/jair2014-geoloc/
    latlon_country = {}
    with open('./data/han_cook_baldwin.geo', 'r') as fin:
        for line in fin:
            line = line.strip()
            fields = line.split('\t')
            country = fields[0].split('-')[-1].upper()
            lat = float(fields[2])
            lon = float(fields[3])
            latlon_country[(lat, lon)] = country
    
    country_count = Counter()
    for point in points:
        country = latlon_country[point]
        country_count[country] += 1
    countries = [c for c, count in country_count.iteritems() if count>100]
    with open('./data/country_count.txt', 'w') as fout:
        json.dump(countries, fout)

def get_state_from_coordinates(coordnates):

    #coordinates = np.array([(34, -118), (40.7, -74)])
    sf = shapefile.Reader('~/datasets/shapefiles/us_states_st99/st99_d00')
    
    #sf = shapefile.Reader("./data/states/cb_2015_us_state_20m")
    shapes = sf.shapes()
    #shapes[i].points
    fields = sf.fields
    records = sf.records()
    state_polygons = defaultdict(list)
    for i, record in enumerate(records):
        state = record[5]
        points = shapes[i].points
        poly = shape(shapes[i])
        state_polygons[state].append(poly)
    
    coor_state = OrderedDict()
    for i in range(coordnates.shape[0]):
        lat, lon = coordnates[i]
        for state, polies in state_polygons.iteritems():
            for poly in polies:
                point = Point(lon, lat)
                if poly.contains(point):
                    coor_state[(lat, lon)] = state.lower()
    return coor_state


def contour(coordinates, preds, scores, world=False, filename="contour", do_contour = False, **kwargs):
    #with open('./data/known_preds429200.pkl', 'rb') as fin:
    #    coordinates, preds, scores = pickle.load(fin)
    coordinates, preds, scores = np.array(coordinates), np.array(preds), np.array(scores)
    
    known_states = get_state_from_coordinates(coordinates)
    state_distances = defaultdict(list)
    state_counts = defaultdict(int)
    state_center = {}
    state_state_count = Counter()

    state_abbr = {v.lower():k for k, v in short_state_names.iteritems()}
    for i, state in enumerate(known_states.values()):
        known_state = state_abbr[state]
        d = scores[i]
        state_distances[known_state].append(d)
        state_counts[known_state] += 1

    from matplotlib import rc
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    scores = np.array(scores)
    lllat = 24.396308
    lllon = -124.848974
    urlat =  49.384358
    urlon = -66.885444
    if world:
        lllat = -90
        lllon = -180
        urlat = 90
        urlon = 180
        
    fig = plt.figure(figsize=(5, 4))
    grid_transform = kwargs.get('grid', False)
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    grid_interpolation_method = 'nearest'
 
    
   

    
    #scores = np.log(scores)
    
    m = Basemap(llcrnrlat=lllat,
    urcrnrlat=urlat,
    llcrnrlon=lllon,
    urcrnrlon=urlon,
    resolution='i', projection='cyl')

    m.drawmapboundary(fill_color = 'white')
    #m.drawcoastlines(linewidth=0.2)
    m.drawcountries(linewidth=0.1)
    if world:
        m.drawstates(linewidth=0.2, color='lightgray')
    #m.fillcontinents(color='white', lake_color='#0000ff', zorder=2)
    #m.drawrivers(color='#0000ff')
    m.drawlsmask(land_color='whitesmoke',ocean_color="#b0c4de", lakes=True)
    #m.drawcounties()
    shp_info = m.readshapefile('~/datasets/shapefiles/us_states_st99/st99_d00','states',drawbounds=True, zorder=0)
    printed_names = []
    ax = plt.gca()
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    for spine in ax.spines.itervalues(): 
        spine.set_visible(False) 

    state_names_set = set(short_state_names.values())
    mi_index = 0
    wi_index = 0
    for shapedict,state in zip(m.states_info, m.states):
        if world: break
        draw_state_name = True
        if shapedict['NAME'] not in state_names_set: continue
        short_name = short_state_names.keys()[short_state_names.values().index(shapedict['NAME'])]
        if short_name in printed_names and short_name not in ['MI', 'WI']: 
            continue
        if short_name == 'MI':
            if mi_index != 3:
                draw_state_name = False
            mi_index += 1
        if short_name == 'WI':
            if wi_index != 2:
                draw_state_name = False
            wi_index += 1
            
        # center of polygon
        x, y = np.array(state).mean(axis=0)
        hull = ConvexHull(state)
        hull_points = np.array(state)[hull.vertices]
        x, y = hull_points.mean(axis=0)
        if short_name == 'MD':
            y = y - 0.5
            x = x + 0.5
        elif short_name == 'DC':
            y = y + 0.1
        elif short_name == 'MI':
            x = x - 1
        elif short_name == 'RI':
            x = x + 1
            y = y - 1
        #poly = MplPolygon(state,facecolor='lightgray',edgecolor='black')
        #x, y = np.median(np.array(state), axis=0)
        # You have to align x,y manually to avoid overlapping for little states
        if draw_state_name:
            plt.text(x+.1, y, short_name, ha="center", fontsize=4)
            state_center[short_name] = [x + .1, y]
        #ax.add_patch(poly)
        #pdb.set_trace()
        printed_names += [short_name,] 
    aggregate_states = True
    if aggregate_states:
        states = sorted(state_distances.keys())
        xs = [state_center[state][0] for state in states]
        ys = [state_center[state][1] for state in states]
        avg_dist = [int(np.median(state_distances[state])) for state in states]
        counts = [max(int(state_counts[state]), 10) for state in states]
        con = m.scatter(xs, ys, c=avg_dist, s=counts, alpha=0.4, cmap=plt.get_cmap('YlOrRd'))
    else:
        mlon, mlat = m(*(coordinates[:,1], coordinates[:,0]))
        # grid data
        if do_contour:
            numcols, numrows = 1000, 1000
            xi = np.linspace(mlon.min()-1, mlon.max()+1, numcols)
            yi = np.linspace(mlat.min()-1, mlat.max()+1, numrows)
        
            xi, yi = np.meshgrid(xi, yi)
            # interpolate
            x, y, z = mlon, mlat, scores
            #pdb.set_trace()
            #zi = griddata(x, y, z, xi, yi)
            zi = gd(
                (mlon, mlat),
                scores,
                (xi, yi),
                method=grid_interpolation_method, rescale=False)
        
            #Remove the lakes and oceans
            data = maskoceans(xi, yi, zi)
            con = m.contourf(xi, yi, data, cmap=plt.get_cmap('YlOrRd'))
            #con2 = m.scatter(mlon, mlat, c='black', marker='.', s=0.2, alpha=0.05 )
        else:
            cmap=plt.get_cmap('YlOrRd')
            con = m.scatter(mlon, mlat, c=scores, s=1, marker='o', alpha=0.1, cmap=cmap )
    #con = m.contour(xi, yi, data, 3, cmap=plt.get_cmap('YlOrRd'), linewidths=1)
    #con = m.contour(x, y, z, 3, cmap=plt.get_cmap('YlOrRd'), tri=True, linewidths=1)
    #conf = m.contourf(x, y, z, 3, cmap=plt.get_cmap('coolwarm'), tri=True)
    cbar = m.colorbar(con,location='right',pad="3%")
    #plt.setp(cbar.ax.get_yticklabels(), visible=False)
    #cbar.ax.tick_params(axis=u'both', which=u'both',length=0)
    #cbar.ax.set_yticklabels(['low', 'high'])
    #tick_locator = ticker.MaxNLocator(nbins=9)
    #cbar.locator = tick_locator
    #cbar.update_ticks()
    cbar.ax.tick_params(labelsize=6) 
    cbar.ax.xaxis.set_tick_params(pad=0)
    cbar.ax.yaxis.set_tick_params(pad=0)
    cbar.set_label('error in km', size=8, labelpad=1)
    for line in cbar.lines: 
        line.set_linewidth(20)
    
    #read countries for world dataset with more than 100 number of users
    #with open('./data/country_count.json', 'r') as fin:
    #    top_countries = set(json.load(fin))
    top_countries = set()
    world_shp_info = m.readshapefile('~/datasets/shapefiles/CNTR_2014_10M_SH/Data/CNTR_RG_10M_2014','world',drawbounds=False, zorder=100)
    for shapedict,state in zip(m.world_info, m.world):
        if not world:
            if shapedict['CNTR_ID'] not in ['CA', 'MX']: continue
        else:
            if shapedict['CNTR_ID'] in top_countries: continue
        poly = MplPolygon(state,facecolor='gray',edgecolor='gray')
        ax.add_patch(poly)
    #plt.title('term: ' + word )
    plt.tight_layout()
    plt.savefig('./' + filename +  '.pdf', bbox_inches='tight')
    plt.close()
    del m

    
if __name__ == '__main__':
    #retrieve_location_from_coordinates()
    #get_state_from_coordinates(coordnates=None)
    contour(coordinates=None, scores=None, world=False, filename='errormap', do_contour=False)
        
        
    
        
