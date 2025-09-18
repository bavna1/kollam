import numpy as np

def get_sample_kolams():
    """
    Returns a dictionary of sample Kolam patterns for testing and demonstration.
    
    Returns:
        dict: Dictionary of sample patterns with descriptive names
    """
    samples = {}
    
    # Simple Cross Pattern
    samples["Simple Cross"] = create_cross_pattern(15)
    
    # Flower Pattern
    samples["Flower Motif"] = create_flower_pattern(15)
    
    # Geometric Square
    samples["Geometric Square"] = create_square_pattern(15)
    
    # Radial Star
    samples["Radial Star"] = create_star_pattern(15)
    
    # Circular Pattern
    samples["Circular Loop"] = create_circular_pattern(15)
    
    # Traditional Diamond
    samples["Diamond Pattern"] = create_diamond_pattern(15)
    
    # Spiral Pattern
    samples["Spiral Design"] = create_spiral_pattern(15)
    
    # Complex Symmetric
    samples["Complex Symmetric"] = create_complex_symmetric_pattern(15)
    
    return samples

def create_cross_pattern(size):
    """Create a simple cross pattern."""
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    
    # Horizontal line
    for j in range(2, size - 2):
        grid[center, j] = 1
    
    # Vertical line
    for i in range(2, size - 2):
        grid[i, center] = 1
    
    return grid

def create_flower_pattern(size):
    """Create a flower-like pattern with petals."""
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    
    # Center point
    grid[center, center] = 1
    
    # Create 8 petals
    import math
    for angle in np.linspace(0, 2*math.pi, 8, endpoint=False):
        for r in range(1, min(center - 1, 4)):
            x = int(center + r * math.cos(angle))
            y = int(center + r * math.sin(angle))
            
            if 0 <= x < size and 0 <= y < size:
                grid[x, y] = 1
                
                # Add width to petals
                if r == 2:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            grid[nx, ny] = 1
    
    return grid

def create_square_pattern(size):
    """Create a geometric square pattern with inner squares."""
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    
    # Create concentric squares
    for square_size in range(2, center, 2):
        # Top and bottom edges
        for j in range(center - square_size, center + square_size + 1):
            if 0 <= j < size:
                grid[center - square_size, j] = 1
                grid[center + square_size, j] = 1
        
        # Left and right edges
        for i in range(center - square_size, center + square_size + 1):
            if 0 <= i < size:
                grid[i, center - square_size] = 1
                grid[i, center + square_size] = 1
    
    return grid

def create_star_pattern(size):
    """Create a star pattern with radiating lines."""
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    
    # Central point
    grid[center, center] = 1
    
    # Create star rays
    import math
    num_rays = 12
    
    for ray in range(num_rays):
        angle = 2 * math.pi * ray / num_rays
        
        for r in range(1, center - 1):
            x = int(center + r * math.cos(angle))
            y = int(center + r * math.sin(angle))
            
            if 0 <= x < size and 0 <= y < size:
                grid[x, y] = 1
                
                # Add decorative elements at certain distances
                if r % 3 == 0 and r > 1:
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            grid[nx, ny] = 1
    
    return grid

def create_circular_pattern(size):
    """Create concentric circular patterns."""
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    
    import math
    
    # Create multiple concentric circles
    for radius in range(2, center - 1, 2):
        circumference = int(2 * math.pi * radius)
        
        for i in range(circumference):
            angle = 2 * math.pi * i / circumference
            x = int(center + radius * math.cos(angle))
            y = int(center + radius * math.sin(angle))
            
            if 0 <= x < size and 0 <= y < size:
                grid[x, y] = 1
    
    # Add connecting spokes
    for angle in np.linspace(0, 2*math.pi, 8, endpoint=False):
        for r in range(1, center - 1):
            x = int(center + r * math.cos(angle))
            y = int(center + r * math.sin(angle))
            
            if 0 <= x < size and 0 <= y < size:
                grid[x, y] = 1
    
    return grid

def create_diamond_pattern(size):
    """Create a diamond-shaped pattern."""
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    
    # Create diamond outline
    for i in range(size):
        for j in range(size):
            # Diamond condition: |i - center| + |j - center| = constant
            for diamond_size in [3, 5, 7]:
                if diamond_size < center:
                    if abs(i - center) + abs(j - center) == diamond_size:
                        grid[i, j] = 1
    
    # Add internal pattern
    for i in range(center - 2, center + 3):
        for j in range(center - 2, center + 3):
            if abs(i - center) + abs(j - center) <= 2:
                grid[i, j] = 1
    
    return grid

def create_spiral_pattern(size):
    """Create a spiral pattern."""
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    
    import math
    
    # Create an Archimedean spiral
    max_radius = center - 2
    num_points = 100
    
    for i in range(num_points):
        # Spiral equation: r = a * theta
        theta = i * 4 * math.pi / num_points  # Multiple rotations
        r = max_radius * i / num_points
        
        x = int(center + r * math.cos(theta))
        y = int(center + r * math.sin(theta))
        
        if 0 <= x < size and 0 <= y < size:
            grid[x, y] = 1
            
            # Add some width to the spiral
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and i % 3 == 0:
                    grid[nx, ny] = 1
    
    return grid

def create_complex_symmetric_pattern(size):
    """Create a complex pattern with multiple symmetries."""
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    
    import math
    
    # Base radial pattern
    for angle in np.linspace(0, 2*math.pi, 16, endpoint=False):
        for r in range(1, center - 1):
            x = int(center + r * math.cos(angle))
            y = int(center + r * math.sin(angle))
            
            if 0 <= x < size and 0 <= y < size:
                # Create varying intensity along the rays
                if r % 2 == 0 or (r > center//2 and r % 3 == 0):
                    grid[x, y] = 1
    
    # Add circular elements
    for radius in [3, 6]:
        if radius < center:
            circumference = int(2 * math.pi * radius)
            for i in range(0, circumference, 2):  # Skip some points for pattern
                angle = 2 * math.pi * i / circumference
                x = int(center + radius * math.cos(angle))
                y = int(center + radius * math.sin(angle))
                
                if 0 <= x < size and 0 <= y < size:
                    grid[x, y] = 1
    
    # Add corner decorations for full symmetry
    corner_patterns = [
        (2, 2), (2, 3), (3, 2),
        (size-3, 2), (size-3, 3), (size-2, 2),
        (2, size-3), (2, size-2), (3, size-3),
        (size-3, size-3), (size-3, size-2), (size-2, size-3)
    ]
    
    for x, y in corner_patterns:
        if 0 <= x < size and 0 <= y < size:
            grid[x, y] = 1
    
    return grid

def create_traditional_pulli_kolam(size):
    """Create a traditional pulli (dot) Kolam pattern."""
    grid = np.zeros((size, size), dtype=int)
    
    # Create a grid of dots with traditional spacing
    dot_spacing = 3
    start_offset = 2
    
    for i in range(start_offset, size - start_offset, dot_spacing):
        for j in range(start_offset, size - start_offset, dot_spacing):
            grid[i, j] = 1
            
            # Create traditional connecting patterns
            # Connect dots in traditional Kolam style
            if i + dot_spacing < size - start_offset:
                # Vertical connection with curve
                mid_i = i + dot_spacing // 2
                grid[mid_i, j] = 1
                if j > start_offset:
                    grid[mid_i, j-1] = 1
                if j < size - start_offset - 1:
                    grid[mid_i, j+1] = 1
            
            if j + dot_spacing < size - start_offset:
                # Horizontal connection with curve
                mid_j = j + dot_spacing // 2
                grid[i, mid_j] = 1
                if i > start_offset:
                    grid[i-1, mid_j] = 1
                if i < size - start_offset - 1:
                    grid[i+1, mid_j] = 1
    
    return grid

def create_rangoli_pattern(size):
    """Create a Rangoli-style pattern (North Indian variant)."""
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    
    # Create lotus-like pattern
    import math
    
    # Central lotus
    grid[center, center] = 1
    
    # Lotus petals (8-fold symmetry)
    for petal in range(8):
        angle = petal * math.pi / 4
        
        # Each petal has multiple points
        for r in range(1, 5):
            base_x = int(center + r * math.cos(angle))
            base_y = int(center + r * math.sin(angle))
            
            if 0 <= base_x < size and 0 <= base_y < size:
                grid[base_x, base_y] = 1
                
                # Add petal width
                perp_angle = angle + math.pi/2
                for width in [-1, 1]:
                    if r <= 3:  # Only for inner parts of petals
                        px = int(base_x + width * 0.5 * math.cos(perp_angle))
                        py = int(base_y + width * 0.5 * math.sin(perp_angle))
                        
                        if 0 <= px < size and 0 <= py < size:
                            grid[px, py] = 1
    
    # Add outer decorative border
    border_radius = center - 2
    if border_radius > 0:
        for angle in np.linspace(0, 2*math.pi, 32, endpoint=False):
            x = int(center + border_radius * math.cos(angle))
            y = int(center + border_radius * math.sin(angle))
            
            if 0 <= x < size and 0 <= y < size:
                grid[x, y] = 1
    
    return grid

def create_muggu_pattern(size):
    """Create a Muggu-style pattern (Andhra Pradesh variant)."""
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    
    # Create interlocking geometric pattern
    # Characteristic diamond and square combination
    
    # Central diamond
    for i in range(size):
        for j in range(size):
            if abs(i - center) + abs(j - center) == 3:
                grid[i, j] = 1
    
    # Corner squares
    square_size = 2
    corners = [
        (square_size, square_size),
        (square_size, size - square_size - 1),
        (size - square_size - 1, square_size),
        (size - square_size - 1, size - square_size - 1)
    ]
    
    for cx, cy in corners:
        for i in range(cx - square_size//2, cx + square_size//2 + 1):
            for j in range(cy - square_size//2, cy + square_size//2 + 1):
                if 0 <= i < size and 0 <= j < size:
                    grid[i, j] = 1
    
    # Connecting lines
    # Horizontal connectors
    for j in range(4, size - 4):
        grid[center, j] = 1
    
    # Vertical connectors
    for i in range(4, size - 4):
        grid[i, center] = 1
    
    return grid

# Additional utility function to get patterns by region
def get_regional_samples():
    """Get sample patterns organized by region."""
    return {
        "Tamil Nadu": {
            "Traditional Pulli": create_traditional_pulli_kolam(15),
            "Flower Kolam": create_flower_pattern(15),
            "Geometric": create_square_pattern(15)
        },
        "Andhra Pradesh": {
            "Muggu Pattern": create_muggu_pattern(15),
            "Diamond Muggu": create_diamond_pattern(15),
            "Star Muggu": create_star_pattern(15)
        },
        "Karnataka": {
            "Rangavalli Circle": create_circular_pattern(15),
            "Spiral Rangavalli": create_spiral_pattern(15),
            "Cross Rangavalli": create_cross_pattern(15)
        },
        "North India": {
            "Rangoli Lotus": create_rangoli_pattern(15),
            "Symmetric Rangoli": create_complex_symmetric_pattern(15)
        }
    }
