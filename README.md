# Circle Detection Script

A command-line tool for detecting circles in images using image processing techniques.

## Requirements

Ensure you have all dependencies installed:
```sh
pip install -r requirements.txt
```

## Usage

Run the script from the command line:

```sh
python main.py --image path/to/image.jpg --threshold 0.8 --region 20
```

### Arguments:
- `--image` → Path to the input image (required)
- `--threshold` → Detection threshold (required)
- `--region` → Region size for non-maximum suppression (default: 20)

## Project Structure
```
/utilities
  ├── image_circle_handling.py
  ├── imarray.py
main.py
README.md
requirements.txt
```

## Output
The script processes the image, detects circles, and displays the result.
