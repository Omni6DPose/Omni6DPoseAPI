"""
Data types for Object Metafile
==============================
"""

from dataclasses import dataclass
from .rotation import SymLabel
import json


@dataclass
class ObjectTag:
    datatype: str
    """Which split does this object belongs to (`train` or `test`)"""
    sceneChanger: bool
    """whether this object has a relatively large size that can be placed objects on"""
    symmetry: SymLabel
    """Symmetry label."""

    # only for Omni6DPose generation
    materialOptions: "list[str]"

    # only for Omni6DPose generation
    upAxis: "list[str]"

    def __post_init__(self):
        if not isinstance(self.symmetry, SymLabel):
            self.symmetry = SymLabel(**self.symmetry)


@dataclass
class ObjectInfo:
    object_id: str
    """Global unique object id."""

    source: str
    """Source dataset of this object.
    
    - `phocal`: the `PhoCaL <https://paperswithcode.com/dataset/phocal>`_ dataset.
    - `omniobject3d`: the `OmniObject3D <https://paperswithcode.com/dataset/omniobject3d>`_ dataset.
    - `google_scan`: the `Google Scanned Objects <https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research>`_ dataset.
    - `real`: supplementary scanned objects collected from real-world.
    """

    # name of this object (deprecated, use object_id instead)
    name: str

    obj_path: str
    """path to the object mesh file, relative to the source dataset path"""

    tag: ObjectTag
    """See :class:`ObjectTag`"""

    class_label: int
    """class ID of this object"""

    class_name: str
    """class name of this object"""

    dimensions: "list[float]"
    """Bounding box size under object space (not NOCS)"""

    def __post_init__(self):
        if not isinstance(self.tag, ObjectTag):
            self.tag = ObjectTag(**self.tag)


@dataclass
class ClassListItem:
    name: str
    """Name of this class"""

    label: int
    """ID of this class"""

    instance_ids: "list[str]"
    """List of the `object_id` of the objects belonging to this class"""

    stat: dict


@dataclass
class ObjectMetaData:
    class_list: "list[ClassListItem]"
    """The complete list of class"""

    instance_dict: "dict[str, ObjectInfo]"
    """Objects indexed by `object_id`"""

    def __post_init__(self):
        if len(self.class_list) > 0 and not isinstance(
            self.class_list[0], ClassListItem
        ):
            self.class_list = [ClassListItem(**x) for x in self.class_list]
        if len(self.instance_dict) > 0 and not isinstance(
            next(iter(self.instance_dict.values())), ObjectInfo
        ):
            self.instance_dict = {
                k: ObjectInfo(**v) for k, v in self.instance_dict.items()
            }

    @staticmethod
    def load_json(path: str) -> "ObjectMetaData":
        """Load object meta data from json file"""

        with open(path, "r") as f:
            jsondata = json.load(f)

        return ObjectMetaData(**jsondata)
