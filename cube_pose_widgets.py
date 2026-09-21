"""Aspect-preserving camera image with clicks in original image coordinates."""
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import QWidget


class CameraImageView(QWidget):
    image_clicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.message = '等待相机'
        self.setMinimumSize(640, 480)

    def setText(self, message):
        self.message = message
        self.image = QImage()
        self.update()

    def setImage(self, image):
        self.image = image.copy()
        self.update()

    def image_rect(self):
        if self.image.isNull():
            return QRectF()
        scale = min(self.width()/self.image.width(), self.height()/self.image.height())
        w, h = self.image.width()*scale, self.image.height()*scale
        return QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)

    def image_position(self, point):
        rect = self.image_rect()
        if rect.isEmpty() or not rect.contains(point):
            return None
        return ((point.x()-rect.x())*self.image.width()/rect.width(),
                (point.y()-rect.y())*self.image.height()/rect.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#121820'))
        if self.image.isNull():
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, self.message)
        else:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawImage(self.image_rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            point = self.image_position(event.localPos())
            if point is not None:
                self.image_clicked.emit(*point)
