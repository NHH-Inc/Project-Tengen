# Robot box labeler

`robot_box_labeler.py` is a small local Python desktop app for reviewing competition images and correcting robot boxes.

## Start it

From the repository root:

```powershell
python -m pip install -r requirements-labeler.txt
python robot_box_labeler.py "C:\path\to\competition-images"
```

If the folder is omitted, the app opens a folder chooser:

```powershell
python robot_box_labeler.py
```

The folder may contain images directly or in nested folders. Supported image formats are JPG, JPEG, PNG, WEBP, TIFF, BMP, and GIF. **Every displayed image must have annotation data** from either a matching JSON file—`frame.jpg` pairs with `frame.json` (or legacy `frame.boxes.json`)—or the folder's `annotations.json` file. Images without readable annotation data are intentionally skipped.

## Required JSON format

The matching JSON file must be either a list of boxes or an object containing a `boxes`, `objects`, or `detections` list. The most direct form is:

```json
{
  "boxes": [
    {"x": 0.12, "y": 0.30, "w": 0.15, "h": 0.24, "color": "blue"}
  ]
}
```

JSON boxes may use normalized `x`, `y`, `w`, `h`; a pixel `bbox: [x, y, width, height]`; or pixel/normalized `x1`, `y1`, `x2`, `y2`. `color`, `alliance`, `side`, or a string `team` value of `red`/`blue` is recognized for overlay color.

The competition-image export format is also supported. Its one `annotations.json` file contains an `images` list; each row names an image and contains its own `boxes` list. Boxes using `bbox_normalized` and `bbox_xyxy` are displayed and those same fields are updated on save.

## Editing and autosave

- Click a box to select it. White corner handles appear; drag one to stretch that box.
- Click empty space to add a default-size box. Auto mode makes clicks on the left side blue and the right side red.
- Drag anywhere that is not a white resize handle to draw a new box, including over an existing box.
- Press `R`, `B`, or `A` to choose red, blue, or Auto for new boxes. `Recolor selected` applies that mode to the selected box; in Auto it uses the box's center (left = blue, right = red).
- Right-click a box or press `Delete`/`Backspace` to remove it.
- The arrow buttons or keyboard arrow keys move between images.

Every add, resize, recolor, delete, image change, and app close writes immediately through an atomic temporary-file replacement. The corresponding JSON data is updated in place; the tool does not read or create YOLO label files.

New boxes use this normalized shape:

```json
{
  "class_name": "robot",
  "color": "red",
  "x": 0.12,
  "y": 0.30,
  "w": 0.15,
  "h": 0.24
}
```
