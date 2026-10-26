from __future__ import annotations

import sys

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from lie_detector.bio.hrv import clean_rr, spectrum
from lie_detector.config import sessions
from lie_detector.gui.data import CONDITION_LABELS, TOPIC_COLORS, load_session
from lie_detector.gui.video import VideoReader, draw_overlays, face_zoom

WINDOW_S = 8.0
ROLE_COLORS = {"I": "#1565c0", "R": "#2e7d32"}
ROLE_LABELS = {"I": "Психолог", "R": "Респондент"}
TOPIC_LABELS = {"pre": "покой", "intro": "знакомство", "alcohol": "алкоголь", "conflict": "конфликт", "deception": "обман", "outro": "завершение"}
Qt = QtCore.Qt
pg.setConfigOptions(antialias=True, background="w", foreground="#333")


def to_pixmap(img):
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy())


class Timeline(pg.PlotWidget):
    seek = QtCore.Signal(float)

    def __init__(self):
        super().__init__()
        self.setMaximumHeight(90)
        self.hideAxis("left")
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#000", width=2))
        self.cursor.sigDragged.connect(lambda l: self.seek.emit(l.value()))
        self.scene().sigMouseClicked.connect(self._click)

    def set_session(self, d):
        self.clear()
        for a, b, topic in d.topic_spans():
            r = pg.LinearRegionItem((a, b), movable=False, brush=pg.mkBrush(TOPIC_COLORS.get(topic, "#ccc") + "aa"))
            r.setZValue(-10)
            self.addItem(r)
            txt = pg.TextItem(TOPIC_LABELS.get(topic, topic), color="#222", anchor=(0, 0))
            txt.setPos(a, 1.0)
            self.addItem(txt)
        for q in d.question_starts():
            self.addItem(pg.InfiniteLine(q, angle=90, pen=pg.mkPen("#1565c0", width=1)))
        if d.stress is not None:
            self.plot(d.stress.t, d.stress.y / 100, pen=pg.mkPen("#c62828", width=1))
        self.addItem(self.cursor)
        self.setXRange(0, d.duration, padding=0)
        self.setYRange(0, 1.1)

    def _click(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.seek.emit(self.plotItem.vb.mapSceneToView(ev.scenePos()).x())

    def set_time(self, t):
        self.cursor.blockSignals(True)
        self.cursor.setValue(t)
        self.cursor.blockSignals(False)


class Transcript(QtWidgets.QListWidget):
    seek = QtCore.Signal(float)

    def __init__(self):
        super().__init__()
        self.setWordWrap(True)
        self.itemClicked.connect(lambda it: self.seek.emit(it.data(Qt.ItemDataRole.UserRole)))
        self._current = None

    def set_session(self, d):
        self.clear()
        for u in d.dialog:
            m, s = divmod(int(u["start"]), 60)
            tag = f"  [{CONDITION_LABELS.get(u.get('condition'), '')}]" if u["role"] == "R" else ""
            it = QtWidgets.QListWidgetItem(f"{m:02d}:{s:02d}  {ROLE_LABELS.get(u['role'], u['role'])}: {u['text']}{tag}")
            it.setForeground(QtGui.QColor(ROLE_COLORS.get(u["role"], "#000")))
            it.setData(Qt.ItemDataRole.UserRole, u["start"])
            self.addItem(it)
        self._current = None

    def set_current(self, i):
        if i is None or i == self._current:
            return
        if self._current is not None and self._current < self.count():
            self.item(self._current).setBackground(QtGui.QColor("white"))
        self.item(i).setBackground(QtGui.QColor("#fff59d"))
        self.scrollToItem(self.item(i), QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        self._current = i


class Signals(pg.GraphicsLayoutWidget):
    ROWS = [("ЭКГ", ["ecg"]), ("Пульс", ["rppg", "ppg"]), ("Дыхание", ["belt", "video_breath", "rppg_resp", "riiv", "airflow"]),
            ("КГР", ["gsr"]), ("ЧСС, уд/мин", None)]

    def set_session(self, d):
        self.clear()
        self.plots, self.cursors = [], []
        rows = [(n, k) for n, k in self.ROWS if k is None or any(x in d.signals for x in k)]
        for r, (title, keys) in enumerate(rows):
            p = self.addPlot(row=r, col=0)
            p.setLabel("left", title)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.getAxis("left").setStyle(showValues=keys is None)
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
            p.addLegend(offset=(-5, 2), labelTextSize="8pt", colCount=3)
            if r < len(rows) - 1:
                p.hideAxis("bottom")
            if self.plots:
                p.setXLink(self.plots[0])
            for s in [d.signals[k] for k in keys if k in d.signals] if keys else d.hr.values():
                y = (s.y - np.nanmedian(s.y)) / (np.nanstd(s.y) + 1e-12) if keys else s.y
                p.plot(s.t, y, pen=pg.mkPen(s.color, width=1.2), name=s.label)
            if keys is None:
                p.setYRange(50, 140)
            c = pg.InfiniteLine(angle=90, pen=pg.mkPen("#000", width=1, style=Qt.PenStyle.DashLine))
            p.addItem(c)
            self.plots.append(p)
            self.cursors.append(c)
            self.ci.layout.setRowStretchFactor(r, 1)

    def set_time(self, t):
        if getattr(self, "plots", None):
            self.plots[0].setXRange(t - WINDOW_S, t + WINDOW_S, padding=0)
            for c in self.cursors:
                c.setValue(t)


class Dashboard(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        self.psd = pg.PlotWidget(title="Спектр ВСР (окно 120 с)")
        self.psd.setLabel("bottom", "Гц")
        for ax in ("bottom", "left"):
            self.psd.getAxis(ax).enableAutoSIPrefix(False)
        self.psd_curve = self.psd.plot(pen=pg.mkPen("#d32f2f", width=1.5))
        for lo, hi, col in ((0.003, 0.04, "#e0e0e0"), (0.04, 0.15, "#bbdefb"), (0.15, 0.4, "#c8e6c9")):
            self.psd.addItem(pg.LinearRegionItem((lo, hi), movable=False, brush=pg.mkBrush(col + "88")))
        self.psd.setXRange(0, 0.45)
        self.bars = pg.PlotWidget(title="Ответ: латентность / темп")
        self.bar_item = pg.BarGraphItem(x=[0, 1, 2], height=[0, 0, 0], width=0.6, brushes=["#42a5f5", "#66bb6a", "#ffa726"])
        self.bars.addItem(self.bar_item)
        self.bars.getAxis("bottom").setTicks([[(0, "латентн., с"), (1, "слог/с"), (2, "аномалия")]])
        self.stress_plot = pg.PlotWidget(title="Индекс стресса")
        self.stress_plot.setYRange(0, 100)
        self.stress_curve = self.stress_plot.plot(pen=pg.mkPen("#c62828", width=1.5))
        self.stress_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#000"))
        self.stress_plot.addItem(self.stress_cursor)
        self.info = QtWidgets.QLabel()
        self.info.setMinimumWidth(230)
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.info.setStyleSheet("font-size: 13px;")
        for w, k in ((self.psd, 3), (self.bars, 2), (self.stress_plot, 3), (self.info, 2)):
            lay.addWidget(w, k)
        self.d = self._hrv_t = None

    def set_session(self, d):
        self.d, self._hrv_t = d, None
        if d.stress is not None:
            self.stress_curve.setData(d.stress.t, d.stress.y)
            self.stress_plot.setTitle(d.stress.label)
            self.stress_plot.setXRange(0, d.duration)

    def set_time(self, t):
        d = self.d
        if d is None:
            return
        self.stress_cursor.setValue(t)
        if "ecg" in d.hr and (self._hrv_t is None or abs(t - self._hrv_t) >= 5):
            self._hrv_t = t
            e = d.hr["ecg"]
            rr_t, rr = clean_rr(np.concatenate([[e.t[0] - 60 / e.y[0]], e.t]))
            m = (rr_t >= t - 60) & (rr_t < t + 60)
            if m.sum() > 30:
                f, p = spectrum(rr_t[m], rr[m], t - 60, t + 60)
                self.psd_curve.setData(f[f <= 0.45], p[f <= 0.45])
        lines = [f"<b>Сессия {d.sid}</b> — сценарий {d.meta['scenario']}"]
        u = d.utterance_at(t)
        if u is not None:
            uu = d.dialog[u]
            lines.append(f"Тема: <b>{TOPIC_LABELS.get(uu['topic'], uu['topic'])}</b>  {CONDITION_LABELS.get(uu.get('condition'), '')}")
        h = d.hrv_at(t)
        if h is not None and np.isfinite(h.get("HR_mean", np.nan)):
            lines += [f"ЧСС: {h['HR_mean']:.0f} уд/мин", f"RMSSD: {h['RMSSD']:.0f} мс  SDNN: {h['SDNN']:.0f} мс",
                      f"ИН Баевского: {h['SI_Baevsky']:.0f}", f"LF/HF: {h['LF_HF']:.2f}"]
        a = d.answer_at(t)
        if a is not None:
            g = lambda k: a.get(k, np.nan)
            self.bar_item.setOpts(height=np.nan_to_num([g("latency"), g("syll_rate"), g("anomaly_norm")]))
            lines += [f"Латентность ответа: {g('latency'):.2f} с", f"Темп: {g('syll_rate'):.1f} слог/с, {g('wpm'):.0f} слов/мин", f"F0: {g('f0_mean'):.0f} Гц"]
        if d.stress is not None:
            lines.append(f"<b>{d.stress.label}: {np.interp(t, d.stress.t, d.stress.y):.0f}</b>")
        self.info.setText("<br>".join(lines))


class Media(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        self.video = QtWidgets.QLabel()
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(480, 270)
        self.video.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.face = QtWidgets.QLabel("лицо")
        self.face.setFixedSize(270, 270)
        self.face.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cb_pose, self.cb_face = QtWidgets.QCheckBox("Скелет"), QtWidgets.QCheckBox("Зоны rPPG")
        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Крупный план лица (вместо бокового ракурса)"))
        for w in (self.face, self.cb_pose, self.cb_face):
            side.addWidget(w)
        self.cb_pose.setChecked(True)
        self.cb_face.setChecked(True)
        side.addStretch()
        lay.addWidget(self.video, 4)
        lay.addLayout(side, 1)
        self.reader = self.d = None

    def set_session(self, d):
        self.d, self.reader = d, VideoReader(d.video_path)

    def set_time(self, t):
        if self.reader is None:
            return
        idx = int(round(t * self.reader.fps))
        img = self.reader.frame(idx)
        if img is None:
            return
        kp = sc = None
        if self.d.pose is not None and idx < len(self.d.pose["kpts"]):
            kp, sc = self.d.pose["kpts"][idx], self.d.pose["scores"][idx]
        pm = to_pixmap(draw_overlays(img, kp, sc, self.cb_pose.isChecked(), self.cb_face.isChecked()))
        self.video.setPixmap(pm.scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        z = face_zoom(img, kp, sc, 270)
        if z is not None:
            self.face.setPixmap(to_pixmap(z))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sid=None):
        super().__init__()
        self.setWindowTitle("Мультимодальный анализ: видео · аудио · биосигналы")
        self.resize(1600, 1000)
        self.timeline, self.transcript, self.signals, self.media, self.dashboard = Timeline(), Transcript(), Signals(), Media(), Dashboard()
        self.views = (self.timeline, self.transcript, self.signals, self.media, self.dashboard)
        self.session_box = QtWidgets.QComboBox()
        self.session_box.addItems(list(sessions()))
        self.btn_play, self.btn_prev, self.btn_next = QtWidgets.QPushButton("▶"), QtWidgets.QPushButton("◀ вопрос"), QtWidgets.QPushButton("вопрос ▶")
        self.time_label = QtWidgets.QLabel("00:00.0")
        top = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Сессия:"), self.session_box, self.btn_prev, self.btn_play, self.btn_next, self.time_label):
            top.addWidget(w)
        top.addStretch()
        split = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        for w in self.views:
            split.addWidget(w)
        split.setSizes([70, 130, 260, 330, 210])
        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.addLayout(top)
        v.addWidget(split)
        self.setCentralWidget(central)
        self.player, self.audio = QMediaPlayer(), QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.timer = QtCore.QTimer(interval=40)
        self.timer.timeout.connect(self._tick)
        self.t = 0.0
        self.session_box.currentTextChanged.connect(self.load)
        self.btn_play.clicked.connect(self.toggle)
        self.btn_prev.clicked.connect(lambda: self.jump_question(-1))
        self.btn_next.clicked.connect(lambda: self.jump_question(+1))
        self.timeline.seek.connect(self.seek)
        self.transcript.seek.connect(self.seek)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, self.toggle)
        QtGui.QShortcut(QtGui.QKeySequence("Right"), self, lambda: self.seek(self.t + 1 / 25))
        QtGui.QShortcut(QtGui.QKeySequence("Left"), self, lambda: self.seek(self.t - 1 / 25))
        self.load(sid or self.session_box.currentText())
        if sid:
            self.session_box.setCurrentText(sid)

    def load(self, sid):
        self.player.stop()
        self.timer.stop()
        self.d = load_session(sid)
        self.player.setSource(QtCore.QUrl.fromLocalFile(self.d.audio_path))
        for w in self.views:
            w.set_session(self.d)
        self.seek(0.0)

    def toggle(self):
        playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        if playing:
            self.player.pause()
            self.timer.stop()
        else:
            self.player.setPosition(int(self.t * 1000))
            self.player.play()
            self.timer.start()
        self.btn_play.setText("▶" if playing else "⏸")

    def seek(self, t):
        self.t = float(np.clip(t, 0, self.d.duration))
        self.player.setPosition(int(self.t * 1000))
        self.update_views()

    def jump_question(self, direction):
        qs = np.array(self.d.question_starts())
        cand = qs[qs > self.t + 0.5] if direction > 0 else qs[qs < self.t - 0.5]
        if len(cand):
            self.seek(cand[0] if direction > 0 else cand[-1])

    def _tick(self):
        self.t = self.player.position() / 1000.0
        self.update_views()

    def update_views(self):
        m, s = divmod(self.t, 60)
        self.time_label.setText(f"{int(m):02d}:{s:04.1f}")
        self.transcript.set_current(self.d.utterance_at(self.t))
        for w in (self.timeline, self.signals, self.media, self.dashboard):
            w.set_time(self.t)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(sys.argv[1] if len(sys.argv) > 1 else None)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
