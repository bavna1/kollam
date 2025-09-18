import numpy as np
import cv2
from scipy.ndimage import label, rotate
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class KolamAnalyzer:
    """Analyzes Kolam patterns for mathematical properties and design principles."""
    
    def __init__(self):
        self.symmetry_threshold = 0.85
        
    def analyze_pattern(self, grid):
        """
        Comprehensive analysis of a Kolam pattern.
        
        Args:
            grid (np.array): Binary grid representing the pattern
            
        Returns:
            dict: Analysis results containing various pattern properties
        """
        results = {}
        
        # Basic pattern properties
        results.update(self._analyze_basic_properties(grid))
        
        # Symmetry analysis
        results.update(self._analyze_symmetry(grid))
        
        # Topological properties
        results.update(self._analyze_topology(grid))
        
        # Motif detection
        results.update(self._detect_motifs(grid))
        
        return results
    
    def _analyze_basic_properties(self, grid):
        """Analyze basic properties of the pattern."""
        total_points = np.sum(grid)
        total_cells = grid.shape[0] * grid.shape[1]
        
        # Connected components analysis
        labeled_array, num_components = label(grid)
        
        return {
            'pattern_density': total_points / total_cells,
            'total_points': int(total_points),
            'connected_components': int(num_components),
            'grid_size': grid.shape
        }
    
    def _analyze_symmetry(self, grid):
        """Detect various types of symmetry in the pattern."""
        symmetry_results = {}
        
        # Reflection symmetry (horizontal, vertical, diagonal)
        h_symmetric = self._check_reflection_symmetry(grid, axis='horizontal')
        v_symmetric = self._check_reflection_symmetry(grid, axis='vertical')
        d1_symmetric = self._check_reflection_symmetry(grid, axis='diagonal1')
        d2_symmetric = self._check_reflection_symmetry(grid, axis='diagonal2')
        
        reflection_types = []
        if h_symmetric: reflection_types.append('horizontal')
        if v_symmetric: reflection_types.append('vertical')
        if d1_symmetric: reflection_types.append('diagonal1')
        if d2_symmetric: reflection_types.append('diagonal2')
        
        symmetry_results['reflection_symmetry'] = reflection_types
        
        # Rotational symmetry
        rotation_order = self._check_rotational_symmetry(grid)
        symmetry_results['rotational_symmetry'] = 360 // rotation_order if rotation_order > 1 else 0
        symmetry_results['rotation_order'] = rotation_order
        
        # Overall symmetry score
        symmetry_score = len(reflection_types) * 0.25 + (rotation_order - 1) * 0.25
        symmetry_results['symmetry_score'] = min(symmetry_score, 1.0)
        
        return symmetry_results
    
    def _check_reflection_symmetry(self, grid, axis):
        """Check if pattern has reflection symmetry along specified axis."""
        if axis == 'horizontal':
            reflected = np.flipud(grid)
        elif axis == 'vertical':
            reflected = np.fliplr(grid)
        elif axis == 'diagonal1':
            reflected = np.transpose(grid)
        elif axis == 'diagonal2':
            reflected = np.transpose(np.flipud(np.fliplr(grid)))
        else:
            return False
        
        # Calculate similarity
        if grid.shape != reflected.shape:
            return False
            
        similarity = np.sum(grid == reflected) / grid.size
        return similarity >= self.symmetry_threshold
    
    def _check_rotational_symmetry(self, grid):
        """Check for rotational symmetry and return the order."""
        max_order = 8  # Check up to 8-fold symmetry
        
        for order in range(2, max_order + 1):
            angle = 360 / order
            is_symmetric = True
            
            for i in range(1, order):
                rotated = rotate(grid.astype(float), angle * i, reshape=False, order=0)
                rotated = (rotated > 0.5).astype(int)
                
                if rotated.shape != grid.shape:
                    is_symmetric = False
                    break
                    
                similarity = np.sum(grid == rotated) / grid.size
                if similarity < self.symmetry_threshold:
                    is_symmetric = False
                    break
            
            if is_symmetric:
                return order
        
        return 1  # No rotational symmetry
    
    def _analyze_topology(self, grid):
        """Analyze topological properties of the pattern."""
        results = {}
        
        # Find closed loops using contour detection
        grid_uint8 = (grid * 255).astype(np.uint8)
        contours, _ = cv2.findContours(grid_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        closed_loops = 0
        for contour in contours:
            if cv2.contourArea(contour) > 2:  # Filter out tiny contours
                closed_loops += 1
        
        results['closed_loops'] = closed_loops
        
        # Euler characteristic approximation
        # For a 2D pattern: χ = V - E + F (vertices - edges + faces)
        vertices = np.sum(grid)
        edges = self._count_edges(grid)
        faces = closed_loops + 1  # Including the outer infinite face
        
        results['euler_characteristic'] = int(vertices - edges + faces)
        
        return results
    
    def _count_edges(self, grid):
        """Count edges in the pattern by checking neighboring connections."""
        edges = 0
        rows, cols = grid.shape
        
        # Check horizontal edges
        for i in range(rows):
            for j in range(cols - 1):
                if grid[i, j] == 1 and grid[i, j + 1] == 1:
                    edges += 1
        
        # Check vertical edges
        for i in range(rows - 1):
            for j in range(cols):
                if grid[i, j] == 1 and grid[i + 1, j] == 1:
                    edges += 1
        
        return edges
    
    def _detect_motifs(self, grid):
        """Detect repeating motifs in the pattern."""
        results = {'motif_count': 0, 'motifs': []}
        
        # Simple motif detection using template matching
        motif_sizes = [3, 5, 7]  # Different motif sizes to check
        
        for size in motif_sizes:
            if size >= min(grid.shape) // 2:
                continue
                
            motifs_found = self._find_repeating_patterns(grid, size)
            if motifs_found:
                results['motifs'].extend(motifs_found)
                results['motif_count'] += len(motifs_found)
        
        return results
    
    def _find_repeating_patterns(self, grid, motif_size):
        """Find repeating patterns of a given size."""
        motifs = []
        rows, cols = grid.shape
        
        # Extract all possible motifs of the given size
        templates = []
        for i in range(0, rows - motif_size + 1, motif_size // 2):
            for j in range(0, cols - motif_size + 1, motif_size // 2):
                template = grid[i:i+motif_size, j:j+motif_size]
                if np.sum(template) > 0:  # Only consider non-empty templates
                    templates.append((template, i, j))
        
        # Find templates that appear multiple times
        for i, (template, start_i, start_j) in enumerate(templates):
            matches = 1  # Count the template itself
            
            for j, (other_template, other_i, other_j) in enumerate(templates[i+1:], i+1):
                if np.array_equal(template, other_template):
                    matches += 1
            
            if matches >= 2:  # Found a repeating motif
                motifs.append({
                    'pattern': template.tolist(),
                    'size': motif_size,
                    'occurrences': matches,
                    'positions': [(start_i, start_j)]
                })
        
        return motifs
    
    def get_pattern_complexity(self, grid):
        """Calculate a complexity score for the pattern."""
        analysis = self.analyze_pattern(grid)
        
        # Complexity factors
        density_factor = analysis['pattern_density']
        symmetry_factor = 1 - analysis['symmetry_score']  # More symmetry = less complex
        component_factor = min(analysis['connected_components'] / 10, 1.0)
        motif_factor = min(analysis['motif_count'] / 5, 1.0)
        
        complexity = (density_factor + symmetry_factor + component_factor + motif_factor) / 4
        return complexity
    
    def compare_patterns(self, grid1, grid2):
        """Compare two patterns and return similarity metrics."""
        if grid1.shape != grid2.shape:
            return {'error': 'Patterns must have the same dimensions'}
        
        # Direct similarity
        direct_similarity = np.sum(grid1 == grid2) / grid1.size
        
        # Analyze both patterns
        analysis1 = self.analyze_pattern(grid1)
        analysis2 = self.analyze_pattern(grid2)
        
        # Compare analysis results
        symmetry_diff = abs(analysis1['symmetry_score'] - analysis2['symmetry_score'])
        density_diff = abs(analysis1['pattern_density'] - analysis2['pattern_density'])
        component_diff = abs(analysis1['connected_components'] - analysis2['connected_components'])
        
        structural_similarity = 1 - (symmetry_diff + density_diff + component_diff/10) / 3
        
        return {
            'direct_similarity': direct_similarity,
            'structural_similarity': max(structural_similarity, 0),
            'overall_similarity': (direct_similarity + structural_similarity) / 2
        }
