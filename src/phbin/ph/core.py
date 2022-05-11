'''
  core.py

  This code is distributed under the constitution of GNU Lisence.
  (c) Yasutaka Nishida (Toshiba)

  Log of core.py

  2021/7/15 release core.py

                                                           '''
#coding:utf-8
#-------------------------------------------------------------
import cv2, os, glob, shutil, gc, pathlib, sys
import Augmentor
import numpy as np
import homcloud.interface as hc
import homcloud.plotly_3d as p3d
import plotly.graph_objects as go
from tqdm import tqdm
import dill, json
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


def pd_filter(pd, phtree, area):
    pairs_in_mask = area.filter_pairs(pd.pairs())
    pairs_list = []
    for i in range(len(pairs_in_mask)):
        pairs_list.append(
            (pairs_in_mask[i].birth_time(), pairs_in_mask[i].death_time()))

    nodes = []
    for i in range(len(pairs_list)):
        nodes = nodes + phtree.pair_nodes_in_rectangle(
            pairs_list[i][0], pairs_list[i][0], pairs_list[i][1],
            pairs_list[i][1])
        #temp_nodes = phtree.nearest_pair_node(pairs_list[i][0],
        #                                      pairs_list[i][1])

        #nodes.append(temp_nodes)

    return nodes


def reverse_map(nodes,
                img,
                color=(255, 0, 0),
                alpha=0.6,
                birth_poition=(255, 0, 0),
                marker_size=2,
                save_to='reverse.png'):
    mapping = hc.draw_volumes_on_2d_image(nodes,
                                          img,
                                          color=(255, 0, 0),
                                          alpha=0.6,
                                          birth_position=(255, 0, 0),
                                          marker_size=2)
    mapping.save(save_to)


def draw_pi(pi, mesh, save_to='pi.png', dpi=300):
    mesh.histogram_from_vector(pi).plot()
    plt.savefig(save_to, dpi=dpi)


def make_pi(pds,
            mesh_range=(-2, 3),
            bins=32,
            sigma=0.01,
            weight=("atan", 0.01, 3),
            save_to='desc.dill',
            superlevel=True):

    mesh_range = mesh_range
    bins = bins
    sigma = sigma
    weight = weight
    save_to = save_to

    mesh = hc.PIVectorizerMesh(mesh_range,
                               bins,
                               sigma=sigma,
                               weight=weight,
                               superlevel=superlevel)

    pdvects = np.vstack(
        [mesh.vectorize(pds[i]) for i in tqdm(range(len(pds)))])

    desc = pdvects / pdvects.max()

    return {'pi': desc, 'mesh': mesh}


def make_pd_pc_alpha(pointcloud, dim=0):
    if type(pointcloud) is list:
        pds = []

        dirpath = './pds'
        if os.path.exists(dirpath):
            shutil.rmtree('./pds')
        else:
            pass
        os.mkdir('./pds')

        for i in tqdm(range(len(pointcloud))):
            pd = hc.PDList.from_alpha_filtration(pointcloud[i],
                                                 save_to='./pds/%s.pdgm' %
                                                 str(i),
                                                 save_boundary_map=True)
            pds.append(pd.dth_diagram(dim))
    else:
        print('pointcloud must be given list type')
        sys.exit(0)

    return pds


def make_pd_pc_rips(pointcloud, maxdim=2, dim=0):
    if type(pointcloud) is list:

        pds = []

        dirpath = './pds'
        if os.path.exists(dirpath):
            shutil.rmtree('./pds')
        else:
            pass
        os.mkdir('./pds')

        for i in tqdm(range(len(pointcloud))):
            dmatrix = distance_matrix(pointcloud[i], pointcloud[i])
            pd = hc.PDList.from_rips_filtration(dmatrix,
                                                maxdim=maxdim,
                                                save_to='./pds/%s.pdgm' %
                                                str(i))
            pds.append(pd.dth_diagram(dim))

    else:
        print('pointcloud must be given list type')
        sys.exit(0)
    return pds


def make_pd(np_img, mode='sublevel', dim=0):
    if type(np_img) is list:
        pds = []

        dirpath = './pds'
        if os.path.exists(dirpath):
            shutil.rmtree('./pds')
        else:
            pass
        os.mkdir('./pds')

        for i in tqdm(range(len(np_img))):
            pd = hc.PDList.from_bitmap_levelset(
                hc.distance_transform(np_img[i], signed=True),
                mode=mode,
                save_to='./pds/%s.pdgm' % str(i))
            pds.append(pd.dth_diagram(dim))

    else:
        print('pointcloud must be given list type')
        sys.exit(0)

    return pds


def make_phtrees(np_img, mode='sublevel', dim=0):
    if type(np_img) is list:
        phtrees = []

        dirpath = './pds'
        if os.path.exists(dirpath):
            shutil.rmtree('./pds')
        else:
            pass
        os.mkdir('./pds')

        for i in tqdm(range(len(np_img))):
            pd = hc.BitmapPHTrees.for_bitmap_levelset(
                hc.distance_transform(np_img[i], signed=True),
                mode=mode,
                save_to='./pds/%s.pdgm' % str(i))
            phtrees.append(pd.bitmap_phtrees(dim))

    else:
        print('pointcloud must be given list type')
        sys.exit(0)

    return phtrees


def show_image(img_file):
    img = cv2.imread(img_file, 0)
    plt.imshow(img, cmap='gray')
    plt.show()


def show_pixel_dist(img_file):
    img = cv2.imread(img_file, 0)
    img_sizex = img.shape[1]
    img_sizey = img.shape[0]
    img_data = []
    for j in range(img_sizey):
        img_data.append(img[j][0:img_sizex])
        img_np = np.array(img_data)
    plt.hist(np.ravel(img_np), range=(0, 256), bins=256)
    None


def binarization(img_file, th=128):
    img = cv2.imread(img_file, 0)

    binary_img = np.array(img) < th
    #binary_img = binary_img.astype(np.int) * 255

    return binary_img


def load_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    return img


def write_image(img, save_to):
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.savefig(save_to, bbox_inches='tight', pad_inches=0)


def draw_pd(pd,
            x_range=None,
            x_bins=None,
            y_range=None,
            y_bins=None,
            dpi=300,
            colorbar='log',
            title='title',
            save_to='pd.png'):

    #extract pairs
    pairs = pd.pairs()
    pairs_birth_list = []
    pairs_death_list = []
    for i in range(len(pairs)):
        pairs_birth_list.append(float(pairs[i].birth))
        pairs_death_list.append(float(pairs[i].death))
    x_max = np.max(pairs_birth_list)
    x_min = np.min(pairs_birth_list)
    y_max = np.max(pairs_death_list)
    y_min = np.min(pairs_death_list)

    if x_range == None:
        #x_range = (np.min([x_min, y_min]) * 1.5, np.max([x_max, y_max]) * 1.5)
        #y_range = (np.min([x_min, y_min]) * 1.5, np.max([x_max, y_max]) * 1.5)
        if abs(x_max - x_min) < 0.1:
            if abs(y_max - y_min) < 0.1:
                x_range = (x_min - 1.0, x_max + 1.0)
                y_range = (y_min - 1.0, y_max + 1.0)
            else:
                x_range = (x_min - 1.0, x_max + 1.0)
                y_range = (y_min, y_max)
        elif abs(y_max - y_min) < 0.1:
            x_range = (x_min - 0.5 * abs(x_min), x_max + 0.5 * abs(x_max))
            y_range = (y_min - 1.0, y_max + 1.0)
        else:
            x_range = (x_min - 0.5 * abs(x_min), x_max + 0.5 * abs(x_max))
            y_range = (y_min - 0.5 * abs(y_min), y_max + 0.5 * abs(y_max))
    else:
        x_range = x_range
        y_range = y_range

    if x_bins == None:
        bins = np.max([x_max, y_max]) - np.min([x_min, y_min])
        delta_bins = bins / 50
        x_bins = int(bins / delta_bins)
        y_bins = int(bins / delta_bins)
    else:
        x_bins = x_bins
        y_bins = y_bins

    histogram = hc.HistoSpec(x_range, x_bins, y_range, y_bins).pd_histogram(pd)
    histogram.plot(colorbar={"type": colorbar, "midpoint": 0}, title=title)
    plt.savefig(save_to, dpi=dpi)  #, bbox_inches='tight', pad_inches=0)
