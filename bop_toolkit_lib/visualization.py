# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualization utilities."""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

import random
from scipy.spatial.transform import Rotation

def draw_rect(im, rect, color=(1.0, 1.0, 1.0)):
  """Draws a rectangle on an image.

  :param im: ndarray (uint8) on which the rectangle will be drawn.
  :param rect: Rectangle defined as [x, y, width, height], where [x, y] is the
    top-left corner.
  :param color: Color of the rectangle.
  :return: Image with drawn rectangle.
  """
  if im.dtype != np.uint8:
    raise ValueError('The image must be of type uint8.')

  im_pil = Image.fromarray(im)
  draw = ImageDraw.Draw(im_pil)
  draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
                 outline=tuple([int(c * 255) for c in color]), fill=None)
  del draw
  return np.asarray(im_pil)


def write_text_on_image(im, txt_list, loc=(3, 0), color=(1.0, 1.0, 1.0),
                        size=20):
  """Writes text info on an image.

  :param im: ndarray on which the text info will be written.
  :param txt_list: List of dictionaries, each describing one info line:
    - 'name': Entry name.
    - 'val': Entry value.
    - 'fmt': String format for the value.
  :param loc: Location of the top left corner of the text box.
  :param color: Font color.
  :param size: Font size.
  :return: Image with written text info.
  """
  im_pil = Image.fromarray(im)

  # Load font.
  try:
    font_path = os.path.join(os.path.dirname(__file__), 'droid_sans_mono.ttf')
    font = ImageFont.truetype(font_path, size)
  except IOError:
    misc.log('Warning: Loading a fallback font.')
    font = ImageFont.load_default()

  draw = ImageDraw.Draw(im_pil)
  for info in txt_list:
    if info['name'] != '':
      txt_tpl = '{}:{' + info['fmt'] + '}'
    else:
      txt_tpl = '{}{' + info['fmt'] + '}'
    txt = txt_tpl.format(info['name'], info['val'])
    draw.text(loc, txt, fill=tuple([int(c * 255) for c in color]), font=font)
    text_width, text_height = font.getsize(txt)
    loc = (loc[0], loc[1] + text_height)
  del draw

  return np.array(im_pil)


def depth_for_vis(depth, valid_start=0.2, valid_end=1.0):
  """Transforms depth values from the specified range to [0, 255].

  :param depth: ndarray with a depth image (1 channel).
  :param valid_start: The beginning of the depth range.
  :param valid_end: The end of the depth range.
  :return: Transformed depth image.
  """
  mask = depth > 0
  depth_n = depth.astype(np.float64)
  depth_n[mask] -= depth_n[mask].min()
  depth_n[mask] /= depth_n[mask].max() / (valid_end - valid_start)
  depth_n[mask] += valid_start
  return depth_n


def vis_object_poses(
      poses, K, renderer, rgb=None, depth=None, vis_rgb_path=None,
      vis_depth_diff_path=None, vis_rgb_resolve_visib=False):
  """Visualizes 3D object models in specified poses in a single image.

  Two visualizations are created:
  1. An RGB visualization (if vis_rgb_path is not None).
  2. A Depth-difference visualization (if vis_depth_diff_path is not None).

  :param poses: List of dictionaries, each with info about one pose:
    - 'obj_id': Object ID.
    - 'R': 3x3 ndarray with a rotation matrix.
    - 't': 3x1 ndarray with a translation vector.
    - 'text_info': Info to write at the object (see write_text_on_image).
  :param K: 3x3 ndarray with an intrinsic camera matrix.
  :param renderer: Instance of the Renderer class (see renderer.py).
  :param rgb: ndarray with the RGB input image.
  :param depth: ndarray with the depth input image.
  :param vis_rgb_path: Path to the output RGB visualization.
  :param vis_depth_diff_path: Path to the output depth-difference visualization.
  :param vis_rgb_resolve_visib: Whether to resolve visibility of the objects
    (i.e. only the closest object is visualized at each pixel).
  """
  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

  # Indicators of visualization types.
  vis_rgb = vis_rgb_path is not None
  vis_depth_diff = vis_depth_diff_path is not None

  if vis_rgb and rgb is None:
    raise ValueError('RGB visualization triggered but RGB image not provided.')

  if (vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib)) and depth is None:
    raise ValueError('Depth visualization triggered but D image not provided.')

  # Prepare images for rendering.
  im_size = None
  ren_rgb = None
  ren_rgb_info = None
  ren_depth = None

  if vis_rgb:
    im_size = (rgb.shape[1], rgb.shape[0])
    ren_rgb = np.zeros(rgb.shape, np.uint8)
    ren_rgb_info = np.zeros(rgb.shape, np.uint8)

  if vis_depth_diff:
    if im_size and im_size != (depth.shape[1], depth.shape[0]):
        raise ValueError('The RGB and D images must have the same size.')
    else:
      im_size = (depth.shape[1], depth.shape[0])

  if vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib):
    ren_depth = np.zeros((im_size[1], im_size[0]), np.float32)

  vis_im_rgb = rgb.astype(np.float32)

  # Render the pose estimates one by one.

    #Cuong add change rotation
    rotation = Rotation.from_matrix(pose['R'])
    rotation_additional = Rotation.from_euler('x', random.uniform(0, 15), degrees=True)
    rotation_updated = rotation * rotation_additional
    #pose['R'] = rotation_updated.as_matrix()

    rotation = Rotation.from_matrix(pose['R'])
    rotation_additional = Rotation.from_euler('y', random.uniform(0, 15), degrees=True)
    rotation_updated = rotation * rotation_additional
    #pose['R'] = rotation_updated.as_matrix()

    rotation = Rotation.from_matrix(pose['R'])    
    rotation_additional = Rotation.from_euler('z', random.uniform(0, 15), degrees=True)
    rotation_updated = rotation * rotation_additional
    #pose['R'] = rotation_updated.as_matrix()

    # Rendering.
    ren_out = renderer.render_object(
      pose['obj_id'], pose['R'], pose['t'], fx, fy, cx, cy)

    m_rgb = None
    if vis_rgb:
      m_rgb = ren_out['rgb']

    m_mask = None
    if vis_depth_diff or (vis_rgb and vis_rgb_resolve_visib):
      m_depth = ren_out['depth']

      # Get mask of the surface parts that are closer than the
      # surfaces rendered before.
      visible_mask = np.logical_or(ren_depth == 0, m_depth < ren_depth)
      m_mask = np.logical_and(m_depth != 0, visible_mask)

      ren_depth[m_mask] = m_depth[m_mask].astype(ren_depth.dtype)

    # Combine the RGB renderings.
    if vis_rgb:
      if vis_rgb_resolve_visib:
        #ren_rgb[m_mask] = m_rgb[m_mask].astype(ren_rgb.dtype)
        # Draw circles at corresponding pixels of visible object.
        # Get the indices of non-zero elements in m_mask
        ys, xs = np.nonzero(m_mask)

        # Set the maximum number of circles to be drawn
        num_object_pixels = len(xs)
        max_circles_proportion = 0.2  # Adjust the proportion as desired
        max_circles = int(max_circles_proportion * num_object_pixels)

        # Randomly sample a subset of indices
        indices = np.random.choice(len(xs), size=min(max_circles, len(xs)), replace=False)
        radius = 1  # Set the radius of the circle
        outid = pose['obj_id'] - 1
        color = get_class_colors(outid)  # Set the color of the circle (BGR format)
        thickness = -1  # Set the thickness to -1 for a filled circle

        # Draw circles at corresponding pixels of visible objects
        for idx in indices:
            x, y = xs[idx], ys[idx]
            center = (x, y)
            vis_im_rgb = cv2.circle(vis_im_rgb, center, radius, color, thickness)
      else:
        ren_rgb_f = ren_rgb.astype(np.float32) + m_rgb.astype(np.float32)
        ren_rgb_f[ren_rgb_f > 255] = 255
        ren_rgb = ren_rgb_f.astype(np.uint8)


  # Blend and save the RGB visualization.
  if vis_rgb:
    misc.ensure_dir(os.path.dirname(vis_rgb_path))

    #vis_im_rgb = 0.5 * rgb.astype(np.float32) + \
    #             0.5 * ren_rgb.astype(np.float32) + \
    #             1.0 * ren_rgb_info.astype(np.float32)
    #Cuong
    vis_im_rgb[vis_im_rgb > 255] = 255
    inout.save_im(vis_rgb_path, vis_im_rgb.astype(np.uint8), jpg_quality=95)
    #cv2.imwrite(vis_rgb_path, vis_im_rgb)



def get_class_colors(class_id):
    colordict = {'gray': [128, 128, 128], 'silver': [192, 192, 192], 'black': [0, 0, 0],
                 'maroon': [128, 0, 0], 'red': [255, 0, 0], 'purple': [128, 0, 128], 'fuchsia': [255, 0, 255],
                 'green': [0, 128, 0],
                 'lime': [0, 255, 0], 'olive': [128, 128, 0], 'yellow': [255, 255, 0], 'navy': [0, 0, 128],
                 'blue': [0, 0, 255],
                 'teal': [0, 128, 128], 'aqua': [0, 255, 255], 'orange': [255, 165, 0], 'indianred': [205, 92, 92],
                 'lightcoral': [240, 128, 128], 'salmon': [250, 128, 114], 'darksalmon': [233, 150, 122],
                 'lightsalmon': [255, 160, 122], 'crimson': [220, 20, 60], 'firebrick': [178, 34, 34],
                 'darkred': [139, 0, 0],
                 'pink': [255, 192, 203], 'lightpink': [255, 182, 193], 'hotpink': [255, 105, 180],
                 'deeppink': [255, 20, 147],
                 'mediumvioletred': [199, 21, 133], 'palevioletred': [219, 112, 147], 'coral': [255, 127, 80],
                 'tomato': [255, 99, 71], 'orangered': [255, 69, 0], 'darkorange': [255, 140, 0], 'gold': [255, 215, 0],
                 'lightyellow': [255, 255, 224], 'lemonchiffon': [255, 250, 205],
                 'lightgoldenrodyellow': [250, 250, 210],
                 'papayawhip': [255, 239, 213], 'moccasin': [255, 228, 181], 'peachpuff': [255, 218, 185],
                 'palegoldenrod': [238, 232, 170], 'khaki': [240, 230, 140], 'darkkhaki': [189, 183, 107],
                 'lavender': [230, 230, 250], 'thistle': [216, 191, 216], 'plum': [221, 160, 221],
                 'violet': [238, 130, 238],
                 'orchid': [218, 112, 214], 'magenta': [255, 0, 255], 'mediumorchid': [186, 85, 211],
                 'mediumpurple': [147, 112, 219], 'blueviolet': [138, 43, 226], 'darkviolet': [148, 0, 211],
                 'darkorchid': [153, 50, 204], 'darkmagenta': [139, 0, 139], 'indigo': [75, 0, 130],
                 'slateblue': [106, 90, 205],
                 'darkslateblue': [72, 61, 139], 'mediumslateblue': [123, 104, 238], 'greenyellow': [173, 255, 47],
                 'chartreuse': [127, 255, 0], 'lawngreen': [124, 252, 0], 'limegreen': [50, 205, 50],
                 'palegreen': [152, 251, 152],
                 'lightgreen': [144, 238, 144], 'mediumspringgreen': [0, 250, 154], 'springgreen': [0, 255, 127],
                 'mediumseagreen': [60, 179, 113], 'seagreen': [46, 139, 87], 'forestgreen': [34, 139, 34],
                 'darkgreen': [0, 100, 0], 'yellowgreen': [154, 205, 50], 'olivedrab': [107, 142, 35],
                 'darkolivegreen': [85, 107, 47], 'mediumaquamarine': [102, 205, 170], 'darkseagreen': [143, 188, 143],
                 'lightseagreen': [32, 178, 170], 'darkcyan': [0, 139, 139], 'cyan': [0, 255, 255],
                 'lightcyan': [224, 255, 255],
                 'paleturquoise': [175, 238, 238], 'aquamarine': [127, 255, 212], 'turquoise': [64, 224, 208],
                 'mediumturquoise': [72, 209, 204], 'darkturquoise': [0, 206, 209], 'cadetblue': [95, 158, 160],
                 'steelblue': [70, 130, 180], 'lightsteelblue': [176, 196, 222], 'powderblue': [176, 224, 230],
                 'lightblue': [173, 216, 230], 'skyblue': [135, 206, 235], 'lightskyblue': [135, 206, 250],
                 'deepskyblue': [0, 191, 255], 'dodgerblue': [30, 144, 255], 'cornflowerblue': [100, 149, 237],
                 'royalblue': [65, 105, 225], 'mediumblue': [0, 0, 205], 'darkblue': [0, 0, 139],
                 'midnightblue': [25, 25, 112],
                 'cornsilk': [255, 248, 220], 'blanchedalmond': [255, 235, 205], 'bisque': [255, 228, 196],
                 'navajowhite': [255, 222, 173], 'wheat': [245, 222, 179], 'burlywood': [222, 184, 135],
                 'tan': [210, 180, 140],
                 'rosybrown': [188, 143, 143], 'sandybrown': [244, 164, 96], 'goldenrod': [218, 165, 32],
                 'darkgoldenrod': [184, 134, 11], 'peru': [205, 133, 63], 'chocolate': [210, 105, 30],
                 'saddlebrown': [139, 69, 19],
                 'sienna': [160, 82, 45], 'brown': [165, 42, 42], 'snow': [255, 250, 250], 'honeydew': [240, 255, 240],
                 'mintcream': [245, 255, 250], 'azure': [240, 255, 255], 'aliceblue': [240, 248, 255],
                 'ghostwhite': [248, 248, 255], 'whitesmoke': [245, 245, 245], 'seashell': [255, 245, 238],
                 'beige': [245, 245, 220], 'oldlace': [253, 245, 230], 'floralwhite': [255, 250, 240],
                 'ivory': [255, 255, 240],
                 'antiquewhite': [250, 235, 215], 'linen': [250, 240, 230], 'lavenderblush': [255, 240, 245],
                 'mistyrose': [255, 228, 225], 'gainsboro': [220, 220, 220], 'lightgrey': [211, 211, 211],
                 'darkgray': [169, 169, 169], 'dimgray': [105, 105, 105], 'lightslategray': [119, 136, 153],
                 'slategray': [112, 128, 144], 'darkslategray': [47, 79, 79], 'white': [255, 255, 255]}

    colornames = list(colordict.keys())
    assert (class_id < len(colornames))

    r, g, b = colordict[colornames[class_id]]
    #r, g, b = colordict[colornames[7]]

    return b, g, r  # for OpenCV
