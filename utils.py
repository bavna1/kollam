import numpy as np
from PIL import Image, ImageDraw
import cv2
import io

def grid_to_image(grid, cell_size=20, dot_size=4, line_width=2):
    """
    Convert a grid pattern to a PIL Image.
    
    Args:
        grid (np.array): Binary grid representing the pattern
        cell_size (int): Size of each grid cell in pixels
        dot_size (int): Size of dots in pixels
        line_width (int): Width of connection lines in pixels
        
    Returns:
        PIL.Image: Generated image
    """
    rows, cols = grid.shape
    img_width = cols * cell_size + cell_size
    img_height = rows * cell_size + cell_size
    
    # Create image with white background
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw grid dots
    dot_color = (128, 128, 128)  # Gray
    for i in range(rows):
        for j in range(cols):
            x = j * cell_size + cell_size // 2
            y = i * cell_size + cell_size // 2
            draw.ellipse([x - dot_size//2, y - dot_size//2, 
                         x + dot_size//2, y + dot_size//2], 
                        fill=dot_color)
    
    # Draw pattern connections
    line_color = (0, 0, 128)  # Dark blue
    active_color = (255, 0, 0)  # Red for active points
    
    # Draw connections between adjacent active points
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                x = j * cell_size + cell_size // 2
                y = i * cell_size + cell_size // 2
                
                # Draw active point
                draw.ellipse([x - dot_size, y - dot_size, 
                            x + dot_size, y + dot_size], 
                           fill=active_color)
                
                # Draw horizontal connection
                if j < cols - 1 and grid[i, j + 1] == 1:
                    x2 = (j + 1) * cell_size + cell_size // 2
                    draw.line([x, y, x2, y], fill=line_color, width=line_width)
                
                # Draw vertical connection
                if i < rows - 1 and grid[i + 1, j] == 1:
                    y2 = (i + 1) * cell_size + cell_size // 2
                    draw.line([x, y, x, y2], fill=line_color, width=line_width)
    
    return img

def image_to_grid(image, grid_size):
    """
    Convert an uploaded image to a grid pattern.
    
    Args:
        image (PIL.Image): Input image
        grid_size (int): Size of the output grid
        
    Returns:
        np.array: Binary grid representing the pattern
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to grid size
    image = image.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply threshold to create binary pattern
    threshold = np.mean(img_array) * 0.7  # Adaptive threshold
    binary_grid = (img_array < threshold).astype(int)
    
    # Clean up the pattern - remove isolated pixels
    cleaned_grid = clean_pattern(binary_grid)
    
    return cleaned_grid

def clean_pattern(grid, min_neighbors=1):
    """
    Clean up a pattern by removing isolated pixels and noise.
    
    Args:
        grid (np.array): Binary grid pattern
        min_neighbors (int): Minimum number of neighbors required
        
    Returns:
        np.array: Cleaned binary grid
    """
    rows, cols = grid.shape
    cleaned = grid.copy()
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                # Count neighbors
                neighbors = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbors += grid[ni, nj]
                
                if neighbors < min_neighbors:
                    cleaned[i, j] = 0
    
    return cleaned

def calculate_pattern_metrics(grid):
    """
    Calculate various metrics for a pattern.
    
    Args:
        grid (np.array): Binary grid pattern
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    rows, cols = grid.shape
    total_points = np.sum(grid)
    
    if total_points == 0:
        return {
            'density': 0,
            'compactness': 0,
            'aspect_ratio': 1,
            'coverage': 0
        }
    
    # Density
    density = total_points / (rows * cols)
    
    # Find bounding box
    active_rows, active_cols = np.where(grid == 1)
    if len(active_rows) > 0:
        min_row, max_row = np.min(active_rows), np.max(active_rows)
        min_col, max_col = np.min(active_cols), np.max(active_cols)
        
        bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        compactness = total_points / bbox_area if bbox_area > 0 else 0
        
        aspect_ratio = (max_col - min_col + 1) / (max_row - min_row + 1)
        coverage = bbox_area / (rows * cols)
    else:
        compactness = 0
        aspect_ratio = 1
        coverage = 0
    
    return {
        'density': density,
        'compactness': compactness,
        'aspect_ratio': aspect_ratio,
        'coverage': coverage
    }

def validate_kolam_pattern(grid):
    """
    Validate if a pattern follows basic Kolam rules.
    
    Args:
        grid (np.array): Binary grid pattern
        
    Returns:
        dict: Validation results
    """
    validation = {
        'is_valid': True,
        'issues': [],
        'suggestions': []
    }
    
    # Check for isolated points
    isolated_points = find_isolated_points(grid)
    if len(isolated_points) > 0:
        validation['issues'].append(f"Found {len(isolated_points)} isolated points")
        validation['suggestions'].append("Connect isolated points to main pattern")
    
    # Check connectivity
    components = count_connected_components(grid)
    if components > 3:
        validation['issues'].append(f"Pattern has {components} disconnected components")
        validation['suggestions'].append("Consider connecting components for better flow")
    
    # Check density
    density = np.sum(grid) / grid.size
    if density < 0.1:
        validation['issues'].append("Pattern density is very low")
        validation['suggestions'].append("Add more elements to create a richer pattern")
    elif density > 0.7:
        validation['issues'].append("Pattern density is very high")
        validation['suggestions'].append("Consider simplifying the pattern")
    
    if validation['issues']:
        validation['is_valid'] = False
    
    return validation

def find_isolated_points(grid):
    """Find points that have no neighbors."""
    rows, cols = grid.shape
    isolated = []
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                has_neighbor = False
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if (0 <= ni < rows and 0 <= nj < cols and 
                            grid[ni, nj] == 1):
                            has_neighbor = True
                            break
                    if has_neighbor:
                        break
                
                if not has_neighbor:
                    isolated.append((i, j))
    
    return isolated

def count_connected_components(grid):
    """Count the number of connected components in the pattern."""
    from scipy.ndimage import label
    labeled_array, num_components = label(grid)
    return num_components

def export_pattern_data(grid, analysis_results=None, filename="kolam_pattern.json"):
    """
    Export pattern and analysis data to JSON format.
    
    Args:
        grid (np.array): Pattern grid
        analysis_results (dict): Optional analysis results
        filename (str): Output filename
        
    Returns:
        str: JSON string representation
    """
    import json
    
    data = {
        'pattern': grid.tolist(),
        'grid_size': grid.shape,
        'timestamp': str(np.datetime64('now')),
        'total_points': int(np.sum(grid))
    }
    
    if analysis_results:
        data['analysis'] = analysis_results
    
    # Calculate additional metrics
    data['metrics'] = calculate_pattern_metrics(grid)
    data['validation'] = validate_kolam_pattern(grid)
    
    return json.dumps(data, indent=2)

def import_pattern_data(json_string):
    """
    Import pattern data from JSON format.
    
    Args:
        json_string (str): JSON string containing pattern data
        
    Returns:
        tuple: (grid, analysis_results)
    """
    import json
    
    data = json.loads(json_string)
    grid = np.array(data['pattern'])
    analysis_results = data.get('analysis')
    
    return grid, analysis_results

def create_pattern_thumbnail(grid, size=(64, 64)):
    """
    Create a small thumbnail image of the pattern.
    
    Args:
        grid (np.array): Pattern grid
        size (tuple): Thumbnail size
        
    Returns:
        PIL.Image: Thumbnail image
    """
    img = grid_to_image(grid, cell_size=8, dot_size=1, line_width=1)
    thumbnail = img.resize(size, Image.Resampling.LANCZOS)
    return thumbnail

def detect_pattern_edges(grid):
    """
    Detect edges in the pattern using computer vision techniques.
    
    Args:
        grid (np.array): Binary pattern grid
        
    Returns:
        np.array: Edge detection result
    """
    # Convert to uint8 format
    grid_uint8 = (grid * 255).astype(np.uint8)
    
    # Apply Canny edge detection
    edges = cv2.Canny(grid_uint8, 50, 150)
    
    # Convert back to binary
    return (edges > 0).astype(int)

def smooth_pattern(grid, iterations=1):
    """
    Apply smoothing to the pattern to reduce noise.
    
    Args:
        grid (np.array): Binary pattern grid
        iterations (int): Number of smoothing iterations
        
    Returns:
        np.array: Smoothed pattern grid
    """
    smoothed = grid.copy()
    
    for _ in range(iterations):
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        smoothed_uint8 = (smoothed * 255).astype(np.uint8)
        
        # Opening: erosion followed by dilation
        opened = cv2.morphologyEx(smoothed_uint8, cv2.MORPH_OPEN, kernel)
        
        # Closing: dilation followed by erosion
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        smoothed = (closed > 127).astype(int)
    
    return smoothed

def get_pattern_signature(grid):
    """
    Generate a unique signature for the pattern that can be used for comparison.
    
    Args:
        grid (np.array): Pattern grid
        
    Returns:
        str: Pattern signature hash
    """
    import hashlib
    
    # Normalize the pattern (center it and remove empty borders)
    normalized = normalize_pattern(grid)
    
    # Create hash from the normalized pattern
    pattern_bytes = normalized.tobytes()
    signature = hashlib.md5(pattern_bytes).hexdigest()
    
    return signature

def normalize_pattern(grid):
    """
    Normalize a pattern by centering it and removing empty borders.
    
    Args:
        grid (np.array): Pattern grid
        
    Returns:
        np.array: Normalized pattern grid
    """
    if np.sum(grid) == 0:
        return grid
    
    # Find bounding box
    active_rows, active_cols = np.where(grid == 1)
    min_row, max_row = np.min(active_rows), np.max(active_rows)
    min_col, max_col = np.min(active_cols), np.max(active_cols)
    
    # Extract the minimal bounding box
    cropped = grid[min_row:max_row+1, min_col:max_col+1]
    
    # Pad to make it square
    height, width = cropped.shape
    max_dim = max(height, width)
    
    # Create square grid and center the pattern
    normalized = np.zeros((max_dim, max_dim), dtype=int)
    start_row = (max_dim - height) // 2
    start_col = (max_dim - width) // 2
    
    normalized[start_row:start_row+height, start_col:start_col+width] = cropped
    
    return normalized
