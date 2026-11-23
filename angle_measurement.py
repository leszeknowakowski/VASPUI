"""Angle annotations for three selected atoms in the structure viewer."""

import numpy as np
from vtk import vtkActor, vtkCellArray, vtkPoints, vtkPolyData, vtkPolyDataMapper


def angle_geometry(points):
    """Return degrees, arc points and label position; the second point is the vertex."""
    points = np.asarray(points, dtype=float)
    if points.shape != (3, 3) or not np.all(np.isfinite(points)):
        raise ValueError("The selected atoms must have finite Cartesian coordinates.")
    first, vertex, last = points
    u, v = first - vertex, last - vertex
    length1, length2 = np.linalg.norm(u), np.linalg.norm(v)
    if min(length1, length2) < 1e-10:
        raise ValueError("Cannot measure an angle when an outer atom coincides with the vertex.")
    u, v = u / length1, v / length2
    cosine = np.clip(np.dot(u, v), -1.0, 1.0)
    radians = np.arccos(cosine)
    tangent = v - cosine * u
    if np.linalg.norm(tangent) < 1e-10:
        # Collinear atoms still define a valid 0 or 180 degree angle.
        tangent = np.cross(u, np.eye(3)[np.argmin(np.abs(u))])
    tangent /= np.linalg.norm(tangent)
    radius = 0.3 * min(length1, length2)
    samples = np.linspace(0, radians, 33)
    arc = vertex + radius * (np.cos(samples)[:, None] * u + np.sin(samples)[:, None] * tangent)
    label = vertex + 1.35 * radius * (np.cos(radians / 2) * u + np.sin(radians / 2) * tangent)
    return float(np.degrees(radians)), arc, label


class AngleMeasurement:
    def __init__(self, indices, plotter):
        self.indices = tuple(indices)
        self.plotter = plotter
        self.actor = None
        self.label_actor = None

    def clear(self):
        for attribute in ("actor", "label_actor"):
            actor = getattr(self, attribute)
            if actor is not None:
                self.plotter.remove_actor(actor, render=False)
                setattr(self, attribute, None)

    def update(self, coordinates):
        self.clear()
        coordinates = np.asarray(coordinates, dtype=float)
        if not self.indices or min(self.indices) < 0 or max(self.indices) >= len(coordinates):
            return None
        selected = coordinates[list(self.indices)]
        try:
            angle, arc, label = angle_geometry(selected)
        except ValueError:
            return None
        points = vtkPoints()
        for point in np.vstack((selected, arc)):
            points.InsertNextPoint(*point)
        lines = vtkCellArray()
        for indices in ((0, 1, 2), tuple(range(3, 3 + len(arc)))):
            lines.InsertNextCell(len(indices))
            for index in indices:
                lines.InsertCellPoint(index)
        mesh = vtkPolyData()
        mesh.SetPoints(points)
        mesh.SetLines(lines)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        self.actor = vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetColor(0, 0.8, 0)
        self.actor.GetProperty().SetLineWidth(2)
        self.actor.GetProperty().LightingOff()
        self.actor.PickableOff()
        self.plotter.add_actor(self.actor, reset_camera=False, render=False)
        self.label_actor = self.plotter.add_point_labels(
            [label], [f"{angle:.2f}°"], font_size=24, text_color=(0, 0.8, 0),
            show_points=False, always_visible=True, shape=None)
        return angle
