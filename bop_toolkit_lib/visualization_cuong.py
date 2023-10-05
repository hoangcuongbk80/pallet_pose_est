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


def vertices_reprojection(vertices, R, t, k):
  p = np.matmul(k, np.matmul(R, vertices.T) + t.reshape(-1,1))
  p[0] = p[0] / (p[2] + 1e-5)
  p[1] = p[1] / (p[2] + 1e-5)
  return p[:2].T


def vis_object_poses(
      poses, k_ycbvideo, object_names_ycbvideo, vertex_ycbvideo, 
      K, renderer, rgb=None, rgb_file=None, vis_rgb_path=None):
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
  print('fx fy cx cy: ', fx, fy, cx, cy)
  print(rgb_file)

  # Indicators of visualization types.
  vis_rgb = vis_rgb_path is not None

  if vis_rgb and rgb is None:
    raise ValueError('RGB visualization triggered but RGB image not provided.')

  # Prepare images for rendering.
  im_size = None
  ren_rgb = None
  ren_rgb_info = None
  ren_depth = None

  if vis_rgb:
    im_size = (rgb.shape[1], rgb.shape[0])
    ren_rgb = np.zeros(rgb.shape, np.uint8)
    ren_rgb_info = np.zeros(rgb.shape, np.uint8)

  #Cuong
  image = cv2.imread(rgb_file)
  height, width, _ = image.shape
  confImg = np.copy(image)
  maskImg = np.zeros((height,width), np.uint8)
  contourImg = np.copy(image)
  print("Cuong save: ", vis_rgb_path)

  # Render the pose estimates one by one.
  for pose in poses:

    # show surface reprojection
    maskImg.fill(0)
    outid = pose['obj_id'] - 1
    #print('obj_id: ', outid)
    #print(k_ycbvideo)
    #print(pose['R'])
    
    #Cuong add change rotation
    # Convert the existing rotation to a rotation object
    rotation = Rotation.from_matrix(pose['R'])
    # Apply the additional rotation around the x-axis
    rotation_additional = Rotation.from_euler('x', random.uniform(0, 15), degrees=True)
    rotation_updated = rotation * rotation_additional
    # Update the 'R' key in the pose dictionary
    pose['R'] = rotation_updated.as_matrix()

    vp = vertices_reprojection(1000*vertex_ycbvideo[outid][:], pose['R'], pose['t'], k_ycbvideo)
    vp= vp[::3]
    #print(vp)
    #print(vertex_ycbvideo[outid][:])
    for p in vp:
        if p[0] != p[0] or p[1] != p[1]:  # check nan
            continue
        if p[0] > 640 or p[1] > 480:
            continue
        maskImg = cv2.circle(maskImg, (int(p[0]), int(p[1])), 1, 255, -1)
        confImg = cv2.circle(confImg, (int(p[0]), int(p[1])), 1, get_class_colors(outid), -1, cv2.LINE_AA)

    # fill the holes
    kernel = np.ones((5,5), np.uint8)
    maskImg = cv2.morphologyEx(maskImg, cv2.MORPH_CLOSE, kernel)
    # find contour
    #contours, _ = cv2.findContours(maskImg, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    #_, contours, _ = cv2.findContours(maskImg, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    #contourImg = cv2.drawContours(contourImg, contours, -1, (255, 255, 255), 4, cv2.LINE_AA) # border
    #contourImg = cv2.drawContours(contourImg, contours, -1, get_class_colors(outid), 2, cv2.LINE_AA)

    cv2.imwrite(vis_rgb_path, confImg)

    # Rendering.
    ren_out = renderer.render_object(
      pose['obj_id'], pose['R'], pose['t'], fx, fy, cx, cy)

    m_rgb = None
    if vis_rgb:
      m_rgb = ren_out['rgb']

    m_mask = None

    # Combine the RGB renderings.
    if vis_rgb:
      ren_rgb_f = ren_rgb.astype(np.float32) + m_rgb.astype(np.float32)
      ren_rgb_f[ren_rgb_f > 255] = 255
      ren_rgb = ren_rgb_f.astype(np.uint8)

      # Draw 2D bounding box and write text info.
      obj_mask = np.sum(m_rgb > 0, axis=2)
      ys, xs = obj_mask.nonzero()
      if len(ys):
        # bbox_color = model_color
        # text_color = model_color
        bbox_color = (0.3, 0.3, 0.3)
        text_color = (1.0, 1.0, 1.0)
        text_size = 11

        bbox = misc.calc_2d_bbox(xs, ys, im_size)
        im_size = (obj_mask.shape[1], obj_mask.shape[0])
        ren_rgb_info = draw_rect(ren_rgb_info, bbox, bbox_color)

        if 'text_info' in pose:
          text_loc = (bbox[0] + 2, bbox[1])
          ren_rgb_info = write_text_on_image(
            ren_rgb_info, pose['text_info'], text_loc, color=text_color,
            size=text_size)

  # Blend and save the RGB visualization.
  if vis_rgb:
    misc.ensure_dir(os.path.dirname(vis_rgb_path))

    vis_im_rgb = 0.5 * rgb.astype(np.float32) + \
                 0.5 * ren_rgb.astype(np.float32) + \
                 1.0 * ren_rgb_info.astype(np.float32)
    vis_im_rgb[vis_im_rgb > 255] = 255
    #inout.save_im(vis_rgb_path, vis_im_rgb.astype(np.uint8), jpg_quality=95)

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
