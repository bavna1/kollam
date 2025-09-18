import numpy as np
import random
from scipy.ndimage import rotate
import math

class PatternGenerator:
    """Generates new Kolam patterns based on analysis results or templates."""
    
    def __init__(self):
        self.random_seed = None
        
    def set_seed(self, seed):
        """Set random seed for reproducible generation."""
        self.random_seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_from_analysis(self, analysis_results, grid_size):
        """
        Generate a new pattern based on analysis of an existing pattern.
        
        Args:
            analysis_results (dict): Results from KolamAnalyzer
            grid_size (int): Size of the output grid
            
        Returns:
            np.array: New pattern grid
        """
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Use the analysis to inform generation
        target_density = analysis_results.get('pattern_density', 0.3)
        symmetry_types = analysis_results.get('reflection_symmetry', [])
        rotation_order = analysis_results.get('rotation_order', 1)
        
        # Generate base pattern
        if 'closed_loops' in analysis_results and analysis_results['closed_loops'] > 0:
            grid = self._generate_loop_pattern(grid_size, target_density)
        else:
            grid = self._generate_radial_pattern(grid_size, target_density)
        
        # Apply symmetries based on analysis
        if symmetry_types:
            grid = self._apply_reflection_symmetry(grid, symmetry_types[0])
        
        if rotation_order > 1:
            grid = self._apply_rotational_symmetry(grid, rotation_order)
        
        return self._ensure_connectivity(grid)
    
    def generate_symmetric_pattern(self, grid_size, symmetry_type="reflection"):
        """
        Generate a pattern with specified symmetry.
        
        Args:
            grid_size (int): Size of the output grid
            symmetry_type (str): Type of symmetry ('reflection', 'rotation', 'both')
            
        Returns:
            np.array: Symmetric pattern grid
        """
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        if symmetry_type == "reflection":
            grid = self._generate_reflection_symmetric_pattern(grid_size)
        elif symmetry_type == "rotation":
            grid = self._generate_rotation_symmetric_pattern(grid_size)
        elif symmetry_type == "both":
            grid = self._generate_full_symmetric_pattern(grid_size)
        else:
            grid = self._generate_basic_pattern(grid_size)
        
        return self._ensure_connectivity(grid)
    
    def generate_template_based(self, grid_size, template_type="closed_loop"):
        """
        Generate pattern based on traditional Kolam templates.
        
        Args:
            grid_size (int): Size of the output grid
            template_type (str): Type of template to use
            
        Returns:
            np.array: Pattern based on template
        """
        if template_type == "closed_loop":
            return self._generate_closed_loop_template(grid_size)
        elif template_type == "open_path":
            return self._generate_open_path_template(grid_size)
        elif template_type == "radial":
            return self._generate_radial_template(grid_size)
        elif template_type == "linear":
            return self._generate_linear_template(grid_size)
        else:
            return self._generate_traditional_motif(grid_size)
    
    def _generate_reflection_symmetric_pattern(self, grid_size):
        """Generate a pattern with reflection symmetry."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Generate pattern on one half
        mid = grid_size // 2
        
        # Create some random points in the upper half
        num_points = random.randint(grid_size // 3, grid_size // 2)
        
        for _ in range(num_points):
            i = random.randint(0, mid - 1)
            j = random.randint(0, grid_size - 1)
            grid[i, j] = 1
        
        # Mirror to create symmetry
        grid = self._apply_reflection_symmetry(grid, "horizontal")
        
        return grid
    
    def _generate_rotation_symmetric_pattern(self, grid_size):
        """Generate a pattern with rotational symmetry."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Generate base pattern in one quadrant
        center = grid_size // 2
        radius = center - 2
        
        # Create some points in a sector
        num_points = random.randint(5, 15)
        
        for _ in range(num_points):
            r = random.uniform(2, radius)
            theta = random.uniform(0, math.pi / 2)  # First quadrant only
            
            x = int(center + r * math.cos(theta))
            y = int(center + r * math.sin(theta))
            
            if 0 <= x < grid_size and 0 <= y < grid_size:
                grid[x, y] = 1
        
        # Apply 4-fold rotational symmetry
        grid = self._apply_rotational_symmetry(grid, 4)
        
        return grid
    
    def _generate_full_symmetric_pattern(self, grid_size):
        """Generate pattern with both reflection and rotational symmetry."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Start with a rotation symmetric pattern
        grid = self._generate_rotation_symmetric_pattern(grid_size)
        
        # Add reflection symmetry
        grid = self._apply_reflection_symmetry(grid, "diagonal1")
        
        return grid
    
    def _generate_closed_loop_template(self, grid_size):
        """Generate a traditional closed loop pattern."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        center = grid_size // 2
        
        # Create concentric patterns
        for radius in range(2, center - 1, 2):
            # Create circular/square loop
            for angle in np.linspace(0, 2*math.pi, radius * 4, endpoint=False):
                x = int(center + radius * math.cos(angle))
                y = int(center + radius * math.sin(angle))
                
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[x, y] = 1
        
        # Add connecting elements
        for i in range(center - 2, center + 3):
            for j in range(center - 2, center + 3):
                if abs(i - center) + abs(j - center) <= 2:
                    grid[i, j] = 1
        
        return grid
    
    def _generate_open_path_template(self, grid_size):
        """Generate an open path pattern."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create flowing curves
        center = grid_size // 2
        
        # Generate several curved paths
        for path_id in range(3):
            start_angle = path_id * 2 * math.pi / 3
            
            for t in np.linspace(0, 2*math.pi, 20):
                r = center * 0.7 * (0.5 + 0.3 * math.sin(2*t))
                x = int(center + r * math.cos(t + start_angle))
                y = int(center + r * math.sin(t + start_angle))
                
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[x, y] = 1
        
        return grid
    
    def _generate_radial_template(self, grid_size):
        """Generate a radial pattern template."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        center = grid_size // 2
        
        # Create radial spokes
        num_spokes = 8
        
        for spoke in range(num_spokes):
            angle = spoke * 2 * math.pi / num_spokes
            
            for r in np.linspace(1, center - 1, center):
                x = int(center + r * math.cos(angle))
                y = int(center + r * math.sin(angle))
                
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[x, y] = 1
                    
                # Add some decorative elements
                if r % 3 == 0:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            grid[nx, ny] = 1
        
        return grid
    
    def _generate_linear_template(self, grid_size):
        """Generate a linear pattern template."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create parallel lines with decorative elements
        spacing = 3
        
        for i in range(2, grid_size - 2, spacing):
            for j in range(grid_size):
                grid[i, j] = 1
                
                # Add decorative perpendicular elements
                if j % 4 == 0:
                    for k in range(-1, 2):
                        if 0 <= i + k < grid_size:
                            grid[i + k, j] = 1
        
        return grid
    
    def _generate_traditional_motif(self, grid_size):
        """Generate traditional Kolam motifs."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        center = grid_size // 2
        
        # Create a traditional flower-like motif
        # Central point
        grid[center, center] = 1
        
        # Petals
        petal_length = min(center - 2, 5)
        
        for angle in np.linspace(0, 2*math.pi, 8, endpoint=False):
            for r in range(1, petal_length + 1):
                x = int(center + r * math.cos(angle))
                y = int(center + r * math.sin(angle))
                
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[x, y] = 1
                
                # Add width to petals
                if r > 1:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size and r <= 3:
                            grid[nx, ny] = 1
        
        return grid
    
    def _apply_reflection_symmetry(self, grid, axis):
        """Apply reflection symmetry to a pattern."""
        if axis == "horizontal":
            bottom_half = np.flipud(grid[:grid.shape[0]//2, :])
            grid[grid.shape[0]//2:, :] = bottom_half
        elif axis == "vertical":
            right_half = np.fliplr(grid[:, :grid.shape[1]//2])
            grid[:, grid.shape[1]//2:] = right_half
        elif axis == "diagonal1":
            grid = np.maximum(grid, grid.T)
        elif axis == "diagonal2":
            flipped = np.flipud(np.fliplr(grid))
            grid = np.maximum(grid, flipped.T)
        
        return grid
    
    def _apply_rotational_symmetry(self, grid, order):
        """Apply rotational symmetry to a pattern."""
        angle = 360 / order
        result = grid.copy()
        
        for i in range(1, order):
            rotated = rotate(grid.astype(float), angle * i, reshape=False, order=0)
            rotated = (rotated > 0.5).astype(int)
            result = np.maximum(result, rotated)
        
        return result
    
    def _ensure_connectivity(self, grid):
        """Ensure the pattern has good connectivity properties."""
        # This is a simplified connectivity enhancement
        # In a full implementation, this would use more sophisticated algorithms
        
        # Add connecting points between nearby isolated points
        rows, cols = grid.shape
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if grid[i, j] == 1:
                    # Check if point is isolated
                    neighbors = [
                        grid[i-1, j], grid[i+1, j],
                        grid[i, j-1], grid[i, j+1]
                    ]
                    
                    if sum(neighbors) == 0:
                        # Add connection to nearest point
                        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                # Check if there's a point within 2 steps
                                for step in range(2, 4):
                                    test_i, test_j = i + di * step, j + dj * step
                                    if (0 <= test_i < rows and 0 <= test_j < cols and 
                                        grid[test_i, test_j] == 1):
                                        grid[ni, nj] = 1
                                        break
                                break
        
        return grid
    
    def _generate_basic_pattern(self, grid_size):
        """Generate a basic random pattern."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Add some random points
        num_points = random.randint(grid_size, grid_size * 2)
        
        for _ in range(num_points):
            i = random.randint(0, grid_size - 1)
            j = random.randint(0, grid_size - 1)
            grid[i, j] = 1
        
        return grid
    
    def _generate_loop_pattern(self, grid_size, density):
        """Generate a pattern that forms closed loops."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        center = grid_size // 2
        
        # Create circular loops
        num_loops = max(2, int(density * 10))
        
        for loop_id in range(num_loops):
            radius = 2 + loop_id * 2
            if radius >= center:
                break
                
            circumference = int(2 * math.pi * radius)
            
            for i in range(circumference):
                angle = 2 * math.pi * i / circumference
                x = int(center + radius * math.cos(angle))
                y = int(center + radius * math.sin(angle))
                
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[x, y] = 1
        
        return grid
    
    def _generate_radial_pattern(self, grid_size, density):
        """Generate a radial pattern."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        center = grid_size // 2
        
        num_rays = max(4, int(density * 16))
        
        for ray in range(num_rays):
            angle = 2 * math.pi * ray / num_rays
            max_radius = center - 1
            
            for r in range(1, max_radius):
                x = int(center + r * math.cos(angle))
                y = int(center + r * math.sin(angle))
                
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[x, y] = 1
        
        return grid
