#!/usr/bin/env python3
"""pcd2pgm の切り抜き高さ (h_min / h_max) を GUI で確認しながら PGM を出力する.

入力は pcd2pgm_node を save_all_cloud: true で実行したときの中間点群
(output_dir/cloud_ground.pcd, cloud_nonground.pcd). 地面抽出 (PMF) は再実行しない.
各非地面点の地面からの高さ dz は pcd2pgm_node と同じく,
近傍の地面点 K=12 点の z の平均を地面高さとして求める.

使い方:
  ros2 run pcd2pgm pcd2pgm_gui.py [--config config_pcd2pgm.yaml]
"""
import argparse
import os
import re
import sys

import matplotlib
import numpy as np
import open3d as o3d
import yaml
from PyQt5 import QtCore, QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv
from scipy.spatial import cKDTree

NODE_NAME = 'pcd2pgm_node'
K_GROUND = 12               # pcd2pgm.cpp と同じ
MAX_DISPLAY_POINTS = 600000  # 描画用に間引く点数の上限 (PGM 出力には全点を使う)
SLIDER_MIN, SLIDER_MAX = -1.0, 5.0  # スライダーの範囲 [m] (数値入力は範囲外も可)
SLIDER_SCALE = 100          # スライダーの刻み 0.01 m


def default_config_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('pcd2pgm'), 'config', 'config_pcd2pgm.yaml')
        # symlink-install ならソース側の設定ファイルを指す
        return os.path.realpath(path)
    except Exception:
        return os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'config', 'config_pcd2pgm.yaml'))


def load_params(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data[NODE_NAME]['ros__parameters']


def save_params(config_path, values):
    """コメントを残したまま値だけを書き換える."""
    with open(config_path) as f:
        text = f.read()
    for key, value in values.items():
        text, n = re.subn(rf'^(\s*{key}:\s*)[^#\n]*?(\s*(#.*)?)$',
                          lambda m: f'{m.group(1)}{value}{m.group(2)}', text, count=1, flags=re.M)
        if n == 0:
            raise ValueError(f'{key} が {config_path} に見つかりません')
    with open(config_path, 'w') as f:
        f.write(text)


def compute_height_above_ground(ground, nonground):
    _, idx = cKDTree(ground).query(nonground, k=K_GROUND, workers=-1)
    return nonground[:, 2] - ground[idx, 2].mean(axis=1)


def save_pgm_yaml(points, resolution, min_points_per_cell, output_dir, name):
    """pcd2pgm.cpp の savePgmYaml と同じ方法で PGM / YAML を書き出す."""
    if len(points) == 0:
        raise ValueError('指定した高さの範囲に点がありません')
    min_x, min_y = points[:, 0].min(), points[:, 1].min()
    max_x, max_y = points[:, 0].max(), points[:, 1].max()
    width = max(1, int(np.ceil((max_x - min_x) / resolution)))
    height = max(1, int(np.ceil((max_y - min_y) / resolution)))

    ix = np.floor((points[:, 0] - min_x) / resolution).astype(np.int64)
    iy = np.floor((points[:, 1] - min_y) / resolution).astype(np.int64)
    ok = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    counts = np.bincount(iy[ok] * width + ix[ok], minlength=width * height).reshape(height, width)
    image = np.where(counts >= min_points_per_cell, 0, 254).astype(np.uint8)[::-1]

    os.makedirs(output_dir, exist_ok=True)
    pgm_path = os.path.join(output_dir, name + '.pgm')
    with open(pgm_path, 'wb') as f:
        f.write(f'P5\n{width} {height}\n255\n'.encode('ascii'))
        f.write(image.tobytes())
    yaml_path = os.path.join(output_dir, name + '.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'image: {name}.pgm\n')
        f.write('mode: trinary\n')
        f.write(f'resolution: {resolution:g}\n')
        f.write(f'origin: [{min_x:.6g}, {min_y:.6g}, 0.0]\n')
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.65\n')
        f.write('free_thresh: 0.196\n')
    return pgm_path, yaml_path, image


class Pcd2PgmGui(QtWidgets.QMainWindow):
    def __init__(self, config_path, params, ground, nonground, dz):
        super().__init__()
        self.setWindowTitle('pcd2pgm height slicer')
        self.config_path = config_path
        self.params = params
        self.nonground = nonground
        self.dz = dz

        # 描画用に間引いた点
        rng = np.random.default_rng(0)
        self.disp_idx = (rng.choice(len(nonground), MAX_DISPLAY_POINTS, replace=False)
                         if len(nonground) > MAX_DISPLAY_POINTS else np.arange(len(nonground)))
        g_idx = (rng.choice(len(ground), MAX_DISPLAY_POINTS // 2, replace=False)
                 if len(ground) > MAX_DISPLAY_POINTS // 2 else np.arange(len(ground)))
        self.ground_disp = pv.PolyData(ground[g_idx])
        self.obstacle_actor = None

        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter)
        self.ground_actor = self.plotter.add_mesh(
            self.ground_disp, color='lightgray', point_size=1, opacity=0.3, name='ground')
        self._setup_panel()

        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_view)
        self.update_view()
        self.plotter.view_xy()
        self.plotter.show_axes()

    def _setup_panel(self):
        dock = QtWidgets.QDockWidget('', self)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        panel = QtWidgets.QWidget()
        dock.setWidget(panel)
        layout = QtWidgets.QVBoxLayout(panel)

        layout.addWidget(QtWidgets.QLabel('地面からの高さ [m]'))
        sliders = QtWidgets.QHBoxLayout()
        self.hmin_spin, self.hmin_slider = self._height_control(sliders, 'h_min', float(self.params['h_min']))
        self.hmax_spin, self.hmax_slider = self._height_control(sliders, 'h_max', float(self.params['h_max']))
        layout.addLayout(sliders)

        form = QtWidgets.QFormLayout()
        self.res_spin = QtWidgets.QDoubleSpinBox()
        self.res_spin.setDecimals(3)
        self.res_spin.setRange(0.01, 1.0)
        self.res_spin.setSingleStep(0.01)
        self.res_spin.setValue(float(self.params['resolution']))
        form.addRow('resolution [m]', self.res_spin)
        self.name_edit = QtWidgets.QLineEdit(str(self.params['save_map_name']))
        form.addRow('save_map_name', self.name_edit)
        layout.addLayout(form)

        self.show_ground = QtWidgets.QCheckBox('地面を表示')
        self.show_ground.setChecked(True)
        self.show_ground.toggled.connect(lambda on: (self.ground_actor.SetVisibility(on), self.plotter.render()))
        layout.addWidget(self.show_ground)

        self.count_label = QtWidgets.QLabel()
        layout.addWidget(self.count_label)

        for text, slot in (('Top view', self.plotter.view_xy),
                           ('Convert to PGM', self.on_convert),
                           ('Save params to config', self.on_save_params)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            layout.addWidget(button)
        layout.addWidget(QtWidgets.QLabel(f'config:\n{self.config_path}'))
        layout.addStretch()

    def _height_control(self, parent_layout, label, value):
        box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel(label)
        title.setAlignment(QtCore.Qt.AlignCenter)
        box.addWidget(title)
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-100.0, 100.0)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        box.addWidget(spin)
        slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        slider.setRange(int(SLIDER_MIN * SLIDER_SCALE), int(SLIDER_MAX * SLIDER_SCALE))
        slider.setValue(int(round(value * SLIDER_SCALE)))
        slider.setMinimumHeight(300)
        box.addWidget(slider, alignment=QtCore.Qt.AlignHCenter)
        parent_layout.addLayout(box)

        def from_slider(v):
            spin.blockSignals(True)
            spin.setValue(v / SLIDER_SCALE)
            spin.blockSignals(False)
            self.update_timer.start(30)

        def from_spin(v):
            slider.blockSignals(True)
            slider.setValue(int(round(v * SLIDER_SCALE)))
            slider.blockSignals(False)
            self.update_timer.start(30)

        slider.valueChanged.connect(from_slider)
        spin.valueChanged.connect(from_spin)
        return spin, slider

    def band_mask(self, dz):
        return (dz >= self.hmin_spin.value()) & (dz <= self.hmax_spin.value())

    def update_view(self):
        dz = self.dz[self.disp_idx]
        mask = self.band_mask(dz)
        self.count_label.setText(f'障害物点: {int(self.band_mask(self.dz).sum())} / {len(self.dz)}')
        if self.obstacle_actor is not None:
            self.plotter.remove_actor(self.obstacle_actor, render=False)
            self.obstacle_actor = None
        if mask.any():
            cloud = pv.PolyData(self.nonground[self.disp_idx][mask])
            cloud['dz'] = dz[mask]
            self.obstacle_actor = self.plotter.add_mesh(
                cloud, scalars='dz', cmap='turbo', point_size=2,
                clim=[self.hmin_spin.value(), max(self.hmax_spin.value(), self.hmin_spin.value() + 1e-3)],
                scalar_bar_args={'title': 'dz [m]', 'vertical': True}, reset_camera=False)
        self.plotter.render()

    def on_convert(self):
        try:
            pgm, yml, image = save_pgm_yaml(
                self.nonground[self.band_mask(self.dz)], self.res_spin.value(),
                int(self.params['min_points_per_cell']), self.params['output_dir'],
                self.name_edit.text().strip())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', str(e))
            return
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        plt.figure('PGM')
        plt.clf()
        plt.imshow(image, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show(block=False)
        QtWidgets.QMessageBox.information(self, 'Saved', f'{pgm}\n{yml}')

    def on_save_params(self):
        values = {'h_min': f'{self.hmin_spin.value():.2f}',
                  'h_max': f'{self.hmax_spin.value():.2f}',
                  'resolution': f'{self.res_spin.value():g}',
                  'save_map_name': f'"{self.name_edit.text().strip()}"'}
        try:
            save_params(self.config_path, values)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', str(e))
            return
        QtWidgets.QMessageBox.information(
            self, 'Saved', f'{self.config_path}\n' + '\n'.join(f'{k}: {v}' for k, v in values.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=default_config_path(), help='pcd2pgm の設定ファイル')
    parser.add_argument('--cloud-dir', default=None,
                        help='cloud_ground.pcd / cloud_nonground.pcd のあるディレクトリ (既定: 設定の output_dir)')
    args, qt_args = parser.parse_known_args()

    params = load_params(args.config)
    cloud_dir = args.cloud_dir or params['output_dir']
    paths = [os.path.join(cloud_dir, n) for n in ('cloud_ground.pcd', 'cloud_nonground.pcd')]
    for p in paths:
        if not os.path.isfile(p):
            sys.exit(f'{p} がありません. save_all_cloud: true で pcd2pgm_node を一度実行してください.')

    print('loading clouds ...', flush=True)
    ground, nonground = (np.asarray(o3d.io.read_point_cloud(p).points) for p in paths)
    print(f'ground: {len(ground)}, nonground: {len(nonground)}. computing height above ground ...', flush=True)
    dz = compute_height_above_ground(ground, nonground)

    app = QtWidgets.QApplication([sys.argv[0]] + qt_args)
    window = Pcd2PgmGui(args.config, params, ground, nonground, dz)
    window.resize(1400, 900)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
