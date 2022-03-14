import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import CheckButtons


class PolygonInteractor(object):
    """
    Modified code from
    https://matplotlib.org/2.0.2/examples/event_handling/poly_editor.html
    An polygon editor
    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError(
                "You must first add the polygon to a figure or canvas before defining the interactor"
            )
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)
        self.line = Line2D(
            x,
            y,
            marker="o",
            markerfacecolor="r",
            animated=True,
            markersize=10,
        )
        self.ax.add_line(self.line)

        self._ind = None  # the active vert

        canvas.mpl_connect("draw_event", self.draw_callback)
        canvas.mpl_connect("button_press_event", self.button_press_callback)
        canvas.mpl_connect(
            "button_release_event", self.button_release_callback
        )
        canvas.mpl_connect("motion_notify_event", self.motion_notify_callback)
        self.canvas = canvas

    def get_poly_points(self):
        return np.asarray(self.poly.xy)

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        "this method is called whenever the polygon object is called"
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        "get the index of the vertex under point if within epsilon tolerance"

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        "whenever a mouse button is pressed"
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        "whenever a mouse button is released"
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def motion_notify_callback(self, event):
        "on mouse movement"
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


flags = {"text": False, "qr": False, "lang": False}


def set_(label):
    flags[label] = not flags[label]


def interactive_get_contour(contours, rescaled_image):

    poly = Polygon(
        contours, animated=True, fill=False, color="yellow", linewidth=5
    )
    _, ax = plt.subplots()
    ax.add_patch(poly)
    ax.set_title(
        (
            "Drag the corners of the box to the corners of the document. \n"
            "Close the window when finished."
        )
    )
    p = PolygonInteractor(ax, poly)

    plt.imshow(rescaled_image)
    plt.axis()
    plt.subplots_adjust(left=0.25, bottom=0.1, right=0.95, top=0.95)
    label = ["text", "qr", "lang"]
    activated = [False, False, False]
    axcheck = plt.axes([0.03, 0.4, 0.15, 0.15])
    chxbox = CheckButtons(axcheck, label, activated)
    chxbox.on_clicked(set_)
    plt.show()
    new_points = p.get_poly_points()[:4]
    new_points = np.array([[p] for p in new_points], dtype="int32")
    return flags, new_points.reshape(4, 2)
