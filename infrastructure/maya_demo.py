import csv

import maya.cmds as cmds

# Define the output CSV file path
output_csv = r"C:\Users\giand\OneDrive\Documents\__packages__\_perso\path_tracer\tests\testdata\input_scene.csv"


def get_scene_geometries():
    """Retrieve all geometries in the Maya scene with their attributes."""
    geometries = []

    for obj in cmds.ls(typ="transform"):
        data = {
            "radius": None,
            "colorR": 0.5,
            "colorG": 0.5,
            "colorB": 0.5,
            "reflection": 1.0,
            "roughness": 0.5,
            "intensityR": 1.0,
            "intensityG": 1.0,
            "intensityB": 1.0,
        }

        if cmds.objectType(obj) == "mesh":
            continue

        if "Sphere" in obj or "Light" in obj:
            position = cmds.xform(obj, query=True, translation=True, worldSpace=True)

            data["positionX"] = position[0]
            data["positionY"] = position[1]
            data["positionZ"] = position[2]

            if "Sphere" in obj:
                data["type"] = "Sphere"
                scaleX = cmds.getAttr(f"{obj}.scaleX")
                scaleY = cmds.getAttr(f"{obj}.scaleY")
                scaleZ = cmds.getAttr(f"{obj}.scaleZ")
                data["radius"] = (scaleX + scaleY + scaleZ) / 3.0

                # Add color if a shader is connected
                shading_groups = cmds.listConnections(obj, type="shadingEngine") or []
                if shading_groups:
                    shaders = (
                        cmds.listConnections(f"{shading_groups[0]}.surfaceShader") or []
                    )
                    if shaders and cmds.attributeQuery(
                        "color", node=shaders[0], exists=True
                    ):
                        color = cmds.getAttr(f"{shaders[0]}.color")
                        data.update(
                            {
                                "colorR": color[0],
                                "colorG": color[1],
                                "colorB": color[2],
                            }
                        )

                        reflection = cmds.getAttr(f"{shaders[0]}.specularColor")[0]
                        data["reflection"] = reflection

                        roughness = cmds.getAttr(f"{shaders[0]}.eccentricity")
                        data["roughness"] = roughness

                geometries.append(data)

            else:
                data["type"] = "Light"
                data["intensityR"] = 1.0
                data["intensityG"] = 1.0
                data["intensityB"] = 1.0

                geometries.append(data)

    return geometries


def write_to_csv(geometries, filepath):
    """Write geometry data to a CSV file."""
    headers = [
        "type",
        "positionX",
        "positionY",
        "positionZ",
        "radius",
        "colorR",
        "colorG",
        "colorB",
        "reflection",
        "roughness",
        "intensityR",
        "intensityG",
        "intensityB",
    ]

    with open(filepath, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(geometries)


# Main execution
if __name__ == "__main__":
    scene_geometries = get_scene_geometries()
    write_to_csv(scene_geometries, output_csv)
    print(f"Exported scene geometries to {output_csv}")
