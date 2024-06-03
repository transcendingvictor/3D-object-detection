# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import rc, animation
from matplotlib import animation
from IPython.display import HTML
import os


from nuscenes.nuscenes import NuScenes
from nuscenes import NuScenesExplorer
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility

import numpy as np

currnte_directory = os.getcwd()
data_directory = os.path.join(currnte_directory, "nuscenes_data")
nusc = NuScenes(version='v1.0-mini', dataroot= data_directory, verbose=True)
# for category in nusc.category:
#     print(category['name'])
    
vehicles = []
for i in range(len(nusc.instance)):
    instance = nusc.instance[i]
    category = nusc.get('category', instance['category_token'])
    if 'vehicle.car' in category['name']:
        vehicles.append(i)
        
# for sensor in nusc.sensor:
#     print(sensor['channel'])
    
def belongs_to(anntoken, expected_cam='CAM_FRONT'):
    boxes = []
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    _, boxes, _ = nusc.get_sample_data(sample_record['data'][expected_cam], box_vis_level=BoxVisibility.ANY,
                                                                                            selected_anntokens=[anntoken])

    if len(boxes) == 1:    
            return True
    return False

cam_front_vehicles = []

for i in vehicles:
    instance = nusc.instance[i]
    ann_token = instance['first_annotation_token']
    if belongs_to(ann_token):
        cam_front_vehicles.append(i)
        
# print('Instance IDs: ')

for i in cam_front_vehicles:
    instance = nusc.instance[i]
    first_token = instance['first_annotation_token']
    last_token = instance['last_annotation_token']
    current_token = first_token

    flag = True
    while current_token != last_token:
        if not belongs_to(current_token):
            flag = False
            break
        current_ann = nusc.get('sample_annotation', current_token)
        current_token = current_ann['next']

    # if flag:
    #     print(i, end=' ')
        
def fig2np(fig):
    '''
    Converts matplotlib figure to numpy array
    '''
    canvas = FigureCanvas(fig)
    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return image

def lidar2d(ann_token):
    '''
    Combines bounding box of instance and LIDAR 2D data and returns numpy array of the rendered figure and the box center.
    '''

    # We retrieve annotation record associated with this annotation token
    ann_record = nusc.get('sample_annotation', ann_token)

    # Now, retrieve sample record associated with the sample token of ann_token
    sample_record = nusc.get('sample', ann_record['sample_token'])

    # Now, we get LIDAR metadata from sample_record['data']['LIDAR_TOP'] and
    # retrieve binary file path of LIDAR data using nusc.get_sample_data()
    # We can also pass annotation tokens to visualize instances. 
    # In our case, we have only one instance to visualize.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, _ = nusc.get_sample_data(lidar, selected_anntokens=[ann_token])

    # Declare matplotlib figure and axes for visualization
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    # We now send data_path to retrieve 3D LIDAR data, but we want to project
    # that onto 2D plane. render_height can in-built function of LidarPointCloud
    # does the job for us, we just need to pass data_path and matplotlib axis.
    LidarPointCloud.from_file(data_path).render_height(ax)
    
    # Let's just draw the instace boxes over the renderd LIDAR data.
    for box in boxes:
        c = np.array(nusc_explorer.get_color(box.name)) / 255.0
        box.render(ax, colors=(c, c, c))
    
    # prevent drawing axes
    plt.axis('off')

    # stop drawing
    # %matplotlib agg

    # convert matplotlib figure to numpy array
    img = fig2np(fig)

    return img, box.center


def generate_images_for_animation(instance_id):
    instance = nusc.instance[instance_id]
    first_token = instance['first_annotation_token']
    last_token = instance['last_annotation_token']
    current_token = first_token
 
    imgs = []
    centers = []
    
    while current_token != last_token:
        current_ann = nusc.get('sample_annotation', current_token)
        img, center = lidar2d(current_ann['token'])
        imgs.append(img)
        centers.append(center)

        current_token = current_ann['next']
    return imgs, centers

imgs, centers = generate_images_for_animation(153)


def init():
    img.set_data(imgs[0])
    return (img,)

def animate(i):
    img.set_data(imgs[i])
    return (img,)

fig = plt.figure(figsize=(9,9))
ax = fig.gca()
img = ax.imshow(imgs[0])
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                                         frames=len(imgs), interval=500, blit=True)
# %matplotlib agg
HTML(anim.to_html5_video())