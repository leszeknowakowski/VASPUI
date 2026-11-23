"""Compare orbital ICOHP matrices along a NEB path."""

from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from lobster.icohp_matrix_viewer import (
    IcohpDataset, IcohpMatrixViewer, OrbitalMatrixBuilder, SpinMatrixPlots,
    Spin, QtCore, QtWidgets,
)
from lobster.custom_color_bar import EditableColorBarItem


@dataclass
class NebIcohpImage:
    name: str
    directory: Path
    dataset: object = None
    error: str = ""


class NebIcohpDataset:
    """Match physical bonds across images, independently of LOBSTER bond IDs."""

    def __init__(self, directory, image_dirs=None):
        self.directory = Path(directory)
        if image_dirs is None:
            image_dirs = sorted(
                (path.name for path in self.directory.iterdir()
                 if path.is_dir() and path.name.isdigit()),
                key=lambda name: (int(name), name),
            )
        self.images = []
        for name in image_dirs:
            image = NebIcohpImage(str(name), self.directory / name)
            self.load_image(image, image.directory / "ICOHPLIST.lobster")
            self.images.append(image)
        self.rebuild_bonds()

    @staticmethod
    def load_image(image, filename):
        image.dataset = None
        if not Path(filename).is_file():
            image.error = "Missing ICOHPLIST.lobster"
            return
        try:
            image.dataset = IcohpDataset(str(filename))
            image.error = ""
        except Exception as error:
            image.error = f"Could not load ICOHPLIST: {error}"

    @staticmethod
    def bond_identity(bond_key, metadata):
        atom1, atom2 = metadata.get("atom1"), metadata.get("atom2")
        if atom1 is None or atom2 is None:
            return ("label", str(bond_key)), False
        translation = tuple(int(value) for value in metadata.get("translation", (0, 0, 0)))
        forward = (str(atom1), str(atom2), translation)
        reverse = (str(atom2), str(atom1), tuple(-value for value in translation))
        return ("atoms", min(forward, reverse)), reverse < forward

    def rebuild_bonds(self):
        self.bond_keys = []
        self.references = {}
        self.image_bonds = []
        vmax = 0.0
        for image in self.images:
            matches = {}
            self.image_bonds.append(matches)
            if image.dataset is None:
                continue
            vmax = max(vmax, *(abs(value) for value in image.dataset.color_levels))
            for index in range(image.dataset.bond_count()):
                key, _, metadata = image.dataset.get_bond(index)
                identity, reversed_order = self.bond_identity(key, metadata)
                if identity not in self.references:
                    self.bond_keys.append(identity)
                    self.references[identity] = (image.dataset, index, reversed_order)
                matches[identity] = (index, reversed_order)
        self.color_levels = (-vmax, vmax) if vmax else (-1.0, 1.0)

    def bond_count(self):
        return len(self.bond_keys)

    def get_bond(self, index):
        dataset, bond_index, _ = self.references[self.bond_keys[index]]
        return dataset.get_bond(bond_index)

    def image_bond(self, image_index, bond_index):
        identity = self.bond_keys[bond_index]
        match = self.image_bonds[image_index].get(identity)
        if match is None:
            return None
        index, reversed_order = match
        dataset = self.images[image_index].dataset
        return (*dataset.get_bond(index), reversed_order != self.references[identity][2])

    def matrices(self, bond_index):
        """Use identical orbital order and dimensions for every image and spin."""
        entries, orbitals_1, orbitals_2 = [], [], []
        for image_index in range(len(self.images)):
            bond = self.image_bond(image_index, bond_index)
            if bond is None or not bond[1].get("orbitals"):
                entries.append(None)
                continue
            up, left, right = OrbitalMatrixBuilder.build(bond[1], Spin.up)
            down, _, _ = OrbitalMatrixBuilder.build(bond[1], Spin.down)
            if bond[3]:
                up, down, left, right = up.T, down.T, right, left
            entries.append((up, down, left, right))
            orbitals_1.extend(orbital for orbital in left if orbital not in orbitals_1)
            orbitals_2.extend(orbital for orbital in right if orbital not in orbitals_2)
        aligned = []
        for entry in entries:
            if entry is None:
                aligned.append(None)
                continue
            up, down, left, right = entry
            indexes = np.ix_([orbitals_1.index(value) for value in left],
                             [orbitals_2.index(value) for value in right])
            matrices = []
            for source in (up, down):
                matrix = np.full((len(orbitals_1), len(orbitals_2)), np.nan)
                matrix[indexes] = source
                matrices.append(matrix)
            aligned.append(matrices)
        return aligned, orbitals_1, orbitals_2


class NebSpinMatrixPlots(SpinMatrixPlots):
    """One column per NEB image and one editable color bar for every matrix."""

    def __init__(self, graphics, images, colorbar_graphics=None):
        self.graphics = graphics
        self.show_values = False
        self.interaction_double_click_callback = None
        self.plots = [
            (graphics.addPlot(title=f"Image {image.name} / Spin Up", row=0, col=index),
             graphics.addPlot(title=f"Image {image.name} / Spin Down", row=1, col=index))
            for index, image in enumerate(images)
        ]
        self.colorbar = EditableColorBarItem(
            colorMap=self._default_colormap(), width=30, orientation="vertical",
            interactive=True, label="ICOHP (eV)", colorMapMenu=self._colormap_menu(),
        )
        if colorbar_graphics is None:
            graphics.addItem(self.colorbar, row=0, col=len(images), rowspan=2)
        else:
            colorbar_graphics.addItem(self.colorbar)
        if self.plots:
            first = self.plots[0][0]
            for pair in self.plots:
                for plot in pair:
                    if plot is not first:
                        plot.setXLink(first)
                        plot.setYLink(first)

    def update_images(self, matrices, orbitals_1, orbitals_2, levels, messages):
        images = []
        colormap = self._current_colormap()
        for index, (pair, entry, message) in enumerate(zip(self.plots, matrices, messages)):
            for spin_index, plot in enumerate(pair):
                plot.clear()
                plot.getAxis("bottom").setTicks([
                    [(i + 0.5, orbital) for i, orbital in enumerate(orbitals_1)]])
                plot.getAxis("left").setTicks([
                    [(i + 0.5, orbital) for i, orbital in enumerate(orbitals_2)]])
                plot.setAspectLocked(True)
                if entry is None:
                    text = pg.TextItem(message, anchor=(0.5, 0.5))
                    plot.addItem(text)
                    text.setPos(len(orbitals_1) / 2, len(orbitals_2) / 2)
                    continue
                matrix = entry[spin_index]
                image = self._add_matrix_image(
                    plot, matrix, orbitals_1, orbitals_2,
                    "Spin Up" if spin_index == 0 else "Spin Down", colormap, levels,
                )
                image.set_double_click_callback(
                    lambda x, y, index=index: self.interaction_double_click_callback(
                        index, f"{orbitals_1[x]}-{orbitals_2[y]}"))
                images.append(image)
                if self.show_values:
                    self._add_value_labels(plot, matrix, levels)
        self.colorbar.setImageItem(images)
        self.colorbar.setLevels(levels)
        self.orbital_shape = (len(orbitals_1), len(orbitals_2))
        self.view_all()

    def view_all(self):
        if self.plots:
            width, height = getattr(self, "orbital_shape", (1, 1))
            self.plots[0][0].setRange(xRange=(0, max(width, 1)),
                                     yRange=(0, max(height, 1)), padding=0.05)


class NebIcohpMatrixViewer(IcohpMatrixViewer):
    """Reuse the standalone viewer controls for a synchronized NEB comparison."""

    def __init__(self, directory, image_dirs=None, neb_window=None, parent=None):
        self.neb_window = neb_window
        self.comparison = NebIcohpDataset(directory, image_dirs)
        self.image_highlights = []
        self._updating_plots = False
        self._highlight_active = False
        super().__init__(dataset=self.comparison, default_dir=directory,
                         parent=parent or neb_window, main_window=neb_window,
                         show_structure=False)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowTitle(f"NEB LOBSTER / ICOHP matrices / {directory}")
        self.matrix_plots.colorbar.sigLevelsChanged.connect(self._levels_changed)

    def showEvent(self, event):
        super().showEvent(event)
        self._highlight_active = True
        self.refresh_structure_highlight()

    def closeEvent(self, event):
        self._highlight_active = False
        super().closeEvent(event)

    def _setup_graphics(self):
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setMinimumSize(max(1, len(self.comparison.images)) * 260, 550)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.graphics)
        plot_area = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(plot_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll, 1)
        self.colorbar_graphics = pg.GraphicsLayoutWidget()
        self.colorbar_graphics.setFixedWidth(110)
        layout.addWidget(self.colorbar_graphics)
        self.main_layout.insertWidget(1, plot_area)
        if self.main_window is None:
            self.main_layout.addWidget(self.info_label)
        self.matrix_plots = NebSpinMatrixPlots(self.graphics, self.comparison.images,
                                              self.colorbar_graphics)

    def _connect_signals(self):
        super()._connect_signals()
        self.open_file_action.setText("Open ICOHPLIST for image...")
        self.reload_action = self.menu_bar.actions()[0].menu().addAction("Reload all images")
        self.reload_action.triggered.connect(self.reload_images)

    def _connect_structure_signals(self):
        if self.neb_window is not None:
            self.neb_window.geo_slider.valueChanged.connect(self.refresh_structure_highlight)

    def _levels_changed(self, *_args):
        if not self._updating_plots:
            self.manual_color_levels = tuple(self.matrix_plots.colorbar.levels())
            self._sync_level_inputs()

    def open_file(self):
        names = [image.name for image in self.comparison.images]
        if not names:
            return
        name, accepted = QtWidgets.QInputDialog.getItem(
            self, "NEB image", "Load ICOHPLIST for image", names, 0, False)
        if not accepted:
            return
        index = names.index(name)
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"ICOHPLIST for image {name}", str(self.comparison.images[index].directory),
            "ICOHPLIST files (ICOHPLIST.lobster *.lobster);;All files (*)")
        if filename:
            self.load_file(filename, index)

    def load_file(self, filename, image_index=None):
        if image_index is None:
            directory = Path(filename).resolve().parent
            image_index = next((i for i, image in enumerate(self.comparison.images)
                                if image.directory.resolve() == directory), None)
        if image_index is None:
            QtWidgets.QMessageBox.warning(self, "Unknown image", "Choose a NEB image using File > Open ICOHPLIST for image.")
            return
        image = self.comparison.images[image_index]
        self.comparison.load_image(image, filename)
        self._rebuild_comparison()
        if image.error:
            QtWidgets.QMessageBox.warning(self, "Load failed", image.error)

    def reload_images(self):
        for image in self.comparison.images:
            filename = (image.dataset.filename if image.dataset is not None
                        else image.directory / "ICOHPLIST.lobster")
            self.comparison.load_image(image, filename)
        self._rebuild_comparison()

    def _rebuild_comparison(self):
        previous = (self.comparison.bond_keys[self.current_bond_index]
                    if self.comparison.bond_count() else None)
        self.comparison.rebuild_bonds()
        self.current_bond_index = (self.comparison.bond_keys.index(previous)
                                   if previous in self.comparison.bond_keys else 0)
        self._sync_level_inputs()
        self.update_plot()

    def update_plot(self):
        self._updating_plots = True
        try:
            if self.comparison.bond_count():
                matrices, left, right = self.comparison.matrices(self.current_bond_index)
            else:
                matrices, left, right = [None] * len(self.comparison.images), [], []
            messages, details = [], []
            for index, image in enumerate(self.comparison.images):
                bond = (self.comparison.image_bond(index, self.current_bond_index)
                        if self.comparison.bond_count() else None)
                message = image.error or ("No orbital-resolved data" if bond else "Bond absent in this image")
                messages.append(message)
                if bond:
                    _, data, metadata, _ = bond
                    details.append(f"Image {image.name}: {metadata['atom_pair']}, "
                                   f"{data['length']:.4f} Å; ICOHP up/down: "
                                   f"{self._format_spin_icohp(metadata['total_icohp'], Spin.up)} / "
                                   f"{self._format_spin_icohp(metadata['total_icohp'], Spin.down)}; "
                                   f"equivalent bonds: {data['number_of_bonds']}"
                                   + (f"; {message}" if matrices[index] is None else ""))
                else:
                    details.append(f"Image {image.name}: {message}")
            self.matrix_plots.update_images(matrices, left, right,
                                            self._current_color_levels(), messages)
            self.info_label.setText("<br>".join(escape(text) for text in details)
                                    or "No numbered NEB image directories found")
            if self.comparison.bond_count():
                _, _, metadata = self.comparison.get_bond(self.current_bond_index)
                self.info_label.setText(
                    f"Bond {self.current_bond_index + 1}/{self.comparison.bond_count()} "
                    f"(translation {metadata['translation']})<br>" + self.info_label.text())
            self.previous_button.setEnabled(self.current_bond_index > 0)
            self.next_button.setEnabled(self.current_bond_index < self.comparison.bond_count() - 1)
            self.refresh_structure_highlight()
        finally:
            self._updating_plots = False

    def open_cohpcar_for_interaction(self, image_index, interaction_key):
        bond = self.comparison.image_bond(image_index, self.current_bond_index)
        if bond is None:
            return
        if bond[3]:
            interaction_key = "-".join(reversed(interaction_key.split("-", 1)))
        from lobster.cohpcar_viewer import open_cohpcar_window

        window = open_cohpcar_window(
            default_dir=Path(self.comparison.images[image_index].dataset.filename).parent,
            selected_bond_key=str(bond[0]), selected_interactions=[interaction_key])
        self.cohpcar_plot_windows.append(window)
        window.show()
        window.raise_()
        window.activateWindow()

    def _clear_structure_highlight(self, render=True):
        for renderer, actors in self.image_highlights:
            for actor in actors:
                renderer.RemoveActor(actor)
        self.image_highlights = []
        if render and self.neb_window is not None:
            self.neb_window.widget.GetRenderWindow().Render()

    def refresh_structure_highlight(self, *_args):
        self._clear_structure_highlight(render=False)
        if self.neb_window is None or not self._highlight_active:
            return
        window = self.neb_window
        if not self.comparison.bond_count():
            window.widget.GetRenderWindow().Render()
            return
        for index, image in enumerate(self.comparison.images):
            bond = self.comparison.image_bond(index, self.current_bond_index)
            if bond is None:
                continue
            data = window.neb.image_data[image.name]
            indexes = self._bond_atom_indexes(bond[2], data)
            if indexes is None:
                continue
            if index == 0:
                coords = window.neb.start_stop_positions[0]
            elif index == len(self.comparison.images) - 1:
                coords = window.neb.start_stop_positions[-1]
            else:
                steps = window.neb.neb_positions[index - 1]
                coords = steps[min(window.geo_slider.value(), len(steps) - 1)]
            point1, point2 = np.asarray(coords)[list(indexes)]
            point2 = point2 + self._translation_vector(bond[2], data)
            actors = [self._create_highlight_sphere(point1, 0.725),
                      self._create_highlight_sphere(point2, 0.725),
                      self._create_highlight_bond(point1, point2)]
            renderer = window.plotters[index]
            for actor in actors:
                renderer.AddActor(actor)
            self.image_highlights.append((renderer, actors))
        window.widget.GetRenderWindow().Render()
