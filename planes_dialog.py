"""Plane geometry and the modeless editor used by Modify > Planes."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from vtk import vtkActor, vtkPlaneSource, vtkPolyDataMapper, vtkTransform


@dataclass(eq=False)
class ScenePlane:
    name: str
    source: vtkPlaneSource
    actor: vtkActor
    hkl: tuple = None
    kind: str = None

    def move(self, displacement):
        self.source.SetCenter(np.asarray(self.source.GetCenter()) + displacement)
        self.source.Update()

    def rotate(self, axis, angle):
        center = np.asarray(self.source.GetCenter())
        transform = vtkTransform()
        transform.RotateWXYZ(angle, *axis)
        points = [self.source.GetOrigin(), self.source.GetPoint1(), self.source.GetPoint2()]
        rotated = [center + transform.TransformVector(np.asarray(p) - center) for p in points]
        self.source.SetOrigin(rotated[0])
        self.source.SetPoint1(rotated[1])
        self.source.SetPoint2(rotated[2])
        self.source.Update()


def miller_plane(hkl, lattice, name):
    """Make a plane through the cell center; lattice vectors are stored as rows."""
    indices = np.asarray(hkl, dtype=float)
    if (indices.shape != (3,) or not np.all(np.isfinite(indices))
            or not np.all(indices == np.round(indices)) or not np.any(indices)):
        raise ValueError("Enter three integer Miller indices; h, k and l cannot all be zero.")
    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("The current structure has no valid lattice vectors.")
    try:
        normal = np.linalg.solve(lattice, indices)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The current lattice is singular; a plane cannot be created.") from exc
    normal /= np.linalg.norm(normal)
    center = lattice.sum(axis=0) / 2
    # Choose a stable orthonormal basis and cover the entire projected unit cell.
    u = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    corners = np.asarray(list(product((0, 1), repeat=3))) @ lattice - center
    half_u = np.max(np.abs(corners @ u)) + 2.0
    half_v = np.max(np.abs(corners @ v)) + 2.0
    source = vtkPlaneSource()
    source.SetOrigin(center - half_u * u - half_v * v)
    source.SetPoint1(center + half_u * u - half_v * v)
    source.SetPoint2(center - half_u * u + half_v * v)
    source.Update()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    actor.GetProperty().SetOpacity(1.0)
    # Display the chosen color without shading that varies with the viewing angle.
    actor.GetProperty().LightingOff()
    actor.PickableOff()
    return ScenePlane(name, source, actor, tuple(int(i) for i in indices))


class PlaneAppearanceDialog(QtWidgets.QDialog):
    """Edit the appearance of the plane currently selected in the plane list."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.setWindowTitle("Plane appearance")
        self.resize(380, 200)
        layout = QtWidgets.QVBoxLayout(self)
        self.plane_label = QtWidgets.QLabel()
        layout.addWidget(self.plane_label)
        self.appearance_group = QtWidgets.QGroupBox("Appearance")
        form = QtWidgets.QFormLayout(self.appearance_group)
        self.color_button = QtWidgets.QPushButton()
        self.color_button.setToolTip("Choose the selected plane's color")
        self.color_button.clicked.connect(self.choose_color)
        form.addRow("Color:", self.color_button)
        self.transparency_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_spinbox = QtWidgets.QSpinBox()
        self.transparency_spinbox.setRange(0, 100)
        self.transparency_spinbox.setSuffix("%")
        transparency_layout = QtWidgets.QHBoxLayout()
        transparency_layout.addWidget(self.transparency_slider, 1)
        transparency_layout.addWidget(self.transparency_spinbox)
        form.addRow("Transparency:", transparency_layout)
        self.transparency_slider.valueChanged.connect(self.change_transparency)
        self.transparency_spinbox.valueChanged.connect(self.change_transparency)
        layout.addWidget(self.appearance_group)
        layout.addWidget(QtWidgets.QLabel("0% = opaque; 100% = fully transparent"))
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(self.close)
        layout.addWidget(close)
        self.refresh()

    def refresh(self):
        plane = self.editor.selected_plane()
        self.appearance_group.setEnabled(plane is not None)
        self.plane_label.setText(plane.name if plane is not None else "Select a plane in the Planes window.")
        if plane is None:
            return
        prop = plane.actor.GetProperty()
        color = QtGui.QColor.fromRgbF(*prop.GetColor())
        swatch = QtGui.QPixmap(20, 20)
        swatch.fill(color)
        self.color_button.setIcon(QtGui.QIcon(swatch))
        self.color_button.setText(f"{color.name().upper()} — Choose color…")
        value = int(round((1.0 - prop.GetOpacity()) * 100))
        for widget in (self.transparency_slider, self.transparency_spinbox):
            blocker = QtCore.QSignalBlocker(widget)
            widget.setValue(value)
            del blocker

    def choose_color(self):
        plane = self.editor.selected_plane()
        if plane is None:
            return
        initial = QtGui.QColor.fromRgbF(*plane.actor.GetProperty().GetColor())
        color = QtWidgets.QColorDialog.getColor(initial, self, f"Color — {plane.name}")
        if color.isValid() and plane in self.editor.controls.planes:
            self.editor.controls.set_plane_color(plane, (color.redF(), color.greenF(), color.blueF()))

    def change_transparency(self, value):
        plane = self.editor.selected_plane()
        if plane is not None:
            self.editor.controls.set_plane_transparency(plane, value)


class PlanesDialog(QtWidgets.QDialog):
    def __init__(self, controls):
        super().__init__(controls)
        self.controls = controls
        self.appearance_dialog = None
        self.setWindowTitle("Planes")
        self.resize(620, 660)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("All planes (check to show, uncheck to hide):"))
        self.plane_list = QtWidgets.QListWidget()
        self.plane_list.setMinimumHeight(110)
        layout.addWidget(self.plane_list)
        self.plane_list.currentItemChanged.connect(self.update_selection)
        self.plane_list.itemChanged.connect(self.change_visibility)

        add_group = QtWidgets.QGroupBox("Add plane — Miller indices")
        add_layout = QtWidgets.QHBoxLayout(add_group)
        self.hkl_inputs = []
        for label, value in zip(("h", "k", "l"), (0, 0, 1)):
            add_layout.addWidget(QtWidgets.QLabel(label))
            spin = QtWidgets.QSpinBox()
            spin.setRange(-999, 999)
            spin.setValue(value)
            self.hkl_inputs.append(spin)
            add_layout.addWidget(spin)
        self.add_button = QtWidgets.QPushButton("Add")
        self.add_button.clicked.connect(self.add_plane)
        add_layout.addWidget(self.add_button)
        layout.addWidget(add_group)

        self.selection_group = QtWidgets.QGroupBox("Selected plane")
        selection_layout = QtWidgets.QVBoxLayout(self.selection_group)
        self.details = QtWidgets.QLabel()
        self.details.setWordWrap(True)
        selection_layout.addWidget(self.details)
        move_layout = QtWidgets.QGridLayout()
        move_layout.addWidget(QtWidgets.QLabel("Move step (Å):"), 0, 0)
        self.move_step = self.step_input(0.1, 0.001, 10000, 3)
        move_layout.addWidget(self.move_step, 0, 1, 1, 2)
        self.move_buttons = {}
        for row, (label, axis) in enumerate((('X', (1, 0, 0)), ('Y', (0, 1, 0)),
                                             ('Z', (0, 0, 1)), ('Normal', None)), 1):
            move_layout.addWidget(QtWidgets.QLabel(label), row, 0)
            for col, sign in ((1, -1), (2, 1)):
                button = QtWidgets.QPushButton("−" if sign < 0 else "+")
                button.clicked.connect(lambda _, a=axis, s=sign: self.move_selected(a, s))
                self.move_buttons[label, sign] = button
                move_layout.addWidget(button, row, col)
        transform_layout = QtWidgets.QHBoxLayout()
        transform_layout.addLayout(move_layout)
        rotation_layout = QtWidgets.QGridLayout()
        rotation_layout.addWidget(QtWidgets.QLabel("Rotation step (°):"), 0, 0)
        self.rotation_step = self.step_input(5, 0.1, 360, 1)
        rotation_layout.addWidget(self.rotation_step, 0, 1, 1, 2)
        rotation_layout.setRowMinimumHeight(4, self.move_buttons["Normal", 1].sizeHint().height())
        self.rotate_buttons = {}
        for row, (label, axis) in enumerate(zip("XYZ", np.eye(3)), 1):
            rotation_layout.addWidget(QtWidgets.QLabel(label), row, 0)
            for col, sign in ((1, -1), (2, 1)):
                button = QtWidgets.QPushButton("−" if sign < 0 else "+")
                button.clicked.connect(lambda _, a=axis, s=sign: self.rotate_selected(a, s))
                self.rotate_buttons[label, sign] = button
                rotation_layout.addWidget(button, row, col)
        transform_layout.addLayout(rotation_layout)
        selection_layout.addLayout(transform_layout)
        self.remove_button = QtWidgets.QPushButton("Remove selected plane")
        self.remove_button.clicked.connect(self.remove_selected)
        self.appearance_button = QtWidgets.QPushButton("Color and transparency…")
        self.appearance_button.clicked.connect(self.show_plane_appearance)
        actions_layout = QtWidgets.QHBoxLayout()
        actions_layout.addWidget(self.appearance_button)
        actions_layout.addWidget(self.remove_button)
        selection_layout.addLayout(actions_layout)
        layout.addWidget(self.selection_group)
        hint = QtWidgets.QLabel(
            "New planes pass through the cell center. Moves use Cartesian coordinates; "
            "rotations use Cartesian axes through the plane center.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(self.close)
        layout.addWidget(close)
        controls.planes_changed.connect(self.refresh)
        self.refresh()

    @staticmethod
    def step_input(value, minimum, maximum, decimals):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def selected_plane(self):
        item = self.plane_list.currentItem()
        return item.data(QtCore.Qt.UserRole) if item is not None else None

    def refresh(self):
        selected = self.selected_plane()
        row = self.plane_list.currentRow()
        blocker = QtCore.QSignalBlocker(self.plane_list)
        listed = [self.plane_list.item(i).data(QtCore.Qt.UserRole)
                  for i in range(self.plane_list.count())]
        if listed != self.controls.planes:
            self.plane_list.clear()
            for plane in self.controls.planes:
                item = QtWidgets.QListWidgetItem(plane.name)
                item.setData(QtCore.Qt.UserRole, plane)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                self.plane_list.addItem(item)
        for i, plane in enumerate(self.controls.planes):
            item = self.plane_list.item(i)
            item.setCheckState(QtCore.Qt.Checked if plane.actor.GetVisibility() else QtCore.Qt.Unchecked)
            if plane is selected:
                row = i
        if self.plane_list.count():
            self.plane_list.setCurrentRow(max(0, min(row, self.plane_list.count() - 1)))
        del blocker
        self.update_selection()

    def update_selection(self, *_):
        plane = self.selected_plane()
        self.selection_group.setEnabled(plane is not None)
        if self.appearance_dialog is not None:
            self.appearance_dialog.refresh()
        if plane is None:
            self.details.setText("Select a plane to move or rotate it.")
            return
        center = ", ".join(f"{x:.3f}" for x in plane.source.GetCenter())
        normal = ", ".join(f"{x:.3f}" for x in plane.source.GetNormal())
        indices = f"Initial hkl: {plane.hkl}\n" if plane.hkl else ""
        self.details.setText(f"{plane.name}\n{indices}Center (Å): {center}\nNormal: {normal}")

    def change_visibility(self, item):
        self.controls.set_plane_visibility(item.data(QtCore.Qt.UserRole),
                                           item.checkState() == QtCore.Qt.Checked)

    def add_plane(self):
        try:
            plane = self.controls.add_miller_plane(tuple(spin.value() for spin in self.hkl_inputs))
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot add plane", str(exc))
            return
        for row in range(self.plane_list.count()):
            if self.plane_list.item(row).data(QtCore.Qt.UserRole) is plane:
                self.plane_list.setCurrentRow(row)
                break

    def move_selected(self, axis, sign):
        plane = self.selected_plane()
        if plane is not None:
            direction = plane.source.GetNormal() if axis is None else axis
            self.controls.move_plane(plane, np.asarray(direction) * self.move_step.value() * sign)

    def rotate_selected(self, axis, sign):
        plane = self.selected_plane()
        if plane is not None:
            self.controls.rotate_plane(plane, axis, self.rotation_step.value() * sign)

    def remove_selected(self):
        plane = self.selected_plane()
        if plane is not None:
            self.controls.remove_plane(plane)

    def show_plane_appearance(self):
        if self.selected_plane() is None:
            return
        if self.appearance_dialog is None:
            self.appearance_dialog = PlaneAppearanceDialog(self)
        self.appearance_dialog.refresh()
        self.appearance_dialog.show()
        self.appearance_dialog.raise_()
        self.appearance_dialog.activateWindow()

    def closeEvent(self, event):
        if self.appearance_dialog is not None:
            self.appearance_dialog.close()
        super().closeEvent(event)
