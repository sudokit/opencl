from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
from sim import Sim, Size
from masks import diamond, circle
from pgcolorbar.colorlegend import ColorLegendItem
import pyqtgraph.exporters as pexp
import cv2
from PIL import Image as im
from time import time, sleep

WIDTH, HEIGHT = 256, 256
FPS = 75
EXPORT = True

class MainWindow(QtWidgets.QMainWindow):
  def __init__(self) -> None:
    super().__init__()
    self.sim = Sim(Size(WIDTH, HEIGHT), 3.0, 1.0, 0)
    
    self.mainWidget = QtWidgets.QWidget()
    self.setCentralWidget(self.mainWidget)
    
    self.mainLayout = QtWidgets.QVBoxLayout()
    self.mainWidget.setLayout(self.mainLayout)
    
    self.plot_item = pg.PlotItem()
    
    view_box = self.plot_item.getViewBox()
    view_box.disableAutoRange(pg.ViewBox.XYAxes)
    
    self.image_item = pg.ImageItem()
    self.image_item.setLookupTable(pg.colormap.getFromMatplotlib('jet').getLookupTable())
    self.plot_item.addItem(self.image_item)
    
    self.color_legend_item = ColorLegendItem(
      imageItem=self.image_item,
      showHistogram=True,
      label='Heat distribution')
    self.color_legend_item.setMinimumHeight(100)
    
    self.graphics_layout_widget = pg.GraphicsLayoutWidget()
    self.graphics_layout_widget.addItem(self.plot_item, 0, 0)
    self.graphics_layout_widget.addItem(self.color_legend_item, 0, 1)
    self.mainLayout.addWidget(self.graphics_layout_widget)
    
    self.exporter = pexp.ImageExporter(self.graphics_layout_widget.scene())

    self.paused = True

    cx, cy = WIDTH//2, HEIGHT//2

    # masks = []
    # offsets = [(60, 60), (-60, -60), (60, -60), (-60, 60)]
    # for offset in offsets:
    #   mask = diamond((cx + offset[0], cy + offset[1]), (WIDTH, HEIGHT))
    #   masks.append(mask)
    # [self.sim.addMask(mask, 300.0) for mask in masks]
    
    # circle_mask = circle((cx, cy), (WIDTH, HEIGHT), 25)
    # self.sim.addMask(circle_mask, 999.0)
    
    circle_mask = circle((WIDTH/2, HEIGHT), (WIDTH, HEIGHT), HEIGHT/2)
    self.sim.addMask(circle_mask, -50.0)
    
    circle_mask2 = circle((WIDTH/2, 0), (WIDTH, HEIGHT), HEIGHT/2)
    self.sim.addMask(circle_mask2, 50.0)
    
    circle_mask3 = circle((WIDTH, HEIGHT/2), (WIDTH, HEIGHT), HEIGHT/2)
    self.sim.addMask(circle_mask3, -50.0)
    
    circle_mask4 = circle((0, HEIGHT/2), (WIDTH, HEIGHT), HEIGHT/2)
    self.sim.addMask(circle_mask4, 50.0)


    self.setImage(self.sim.u_curr)
    self.color_legend_item.autoScaleFromImage()

    self.timer = QtCore.QTimer()
    self.timer.timeout.connect(self.update_plot)
    self.timer.start(0)
    
    self.last_time = time()
    self.fps = None

    img = self.exporter.export(toBytes=True)
    width, height = img.width(), img.height()
    self.plot_item.setRange(xRange=[0, self.sim.dims.width], yRange=[0, self.sim.dims.height])
    self.video_writer = cv2.VideoWriter('./out/woop.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, height))
    
  def setImage(self, img: np.ndarray):
    self.image_item.setImage(img.T)
    # self.color_legend_item.autoScaleFromImage()
  
  def keyPressEvent(self, event: QKeyEvent | None) -> None:
    if event.key() == Qt.Key.Key_Space:
      self.paused = not self.paused
    event.accept()

  def update_plot(self):
    if not self.paused:
      self.sim.u_curr = self.sim.u_next
      self.sim.u_next = self.sim.calc(self.sim.u_curr, 18)
      self.setImage(self.sim.u_curr)
      
      if EXPORT:
        img = self.exporter.export(toBytes=True)
        width, height = img.width(), img.height()
        bytes = img.bits().asstring(img.byteCount())
        frame = np.frombuffer(bytes, dtype=np.uint8).reshape(height, width, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        self.video_writer.write(frame)
    
    # fps stuff
    now = time()
    dt = now - self.last_time
    self.last_time = now
    if dt != 0:
      if self.fps is None:
        self.fps = 1.0 / dt
      else:
        s = np.clip(dt * 3.0, 0, 1)
        self.fps = self.fps * (1 - s) + (1.0 / dt) * s
      if self.fps > FPS: sleep(dt)
    self.setWindowTitle(f"{self.fps:.3f} fps | {dt:.4f} dt | {'paused' if self.paused else 'running'}")

  def closeEvent(self, event: QKeyEvent | None):
    print("Released video writer")
    self.video_writer.release()
    event.accept()
    
if __name__ == '__main__':
  app = QtWidgets.QApplication([])
  # loop = QEventLoop(app)
  # asyncio.set_event_loop(loop)
  # main = MainWindow(app)
  # loop.for_ever()
  main = MainWindow()
  main.show()
  app.exec()