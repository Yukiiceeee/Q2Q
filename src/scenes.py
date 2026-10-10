"""Scene identifiers shared by detection, extraction prompts and FQ prompts.

Kept as a dependency-free leaf module so that `src.preprocessing` and
`src.prompts` can both reference the same constants without importing each
other's packages.
"""

SCENE_DIALOGUE = "dialogue"
SCENE_TRAJECTORY = "trajectory"
SCENE_DOCUMENT = "document"

__all__ = ["SCENE_DIALOGUE", "SCENE_TRAJECTORY", "SCENE_DOCUMENT"]
