import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import ListedColormap

class KolamVisualizer:
    """Handles visualization of Kolam patterns and analysis results."""
    
    def __init__(self):
        self.dot_color = '#666666'
        self.line_color = '#000080'
        self.highlight_color = '#FF6B6B'
        self.grid_color = '#CCCCCC'
        
    def create_interactive_grid(self, grid):
        """
        Create an interactive grid visualization for drawing Kolam patterns.
        
        Args:
            grid (np.array): Current pattern grid
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        rows, cols = grid.shape
        
        # Draw grid background
        for i in range(rows + 1):
            ax.axhline(y=i-0.5, color=self.grid_color, linewidth=0.5, alpha=0.5)
        for j in range(cols + 1):
            ax.axvline(x=j-0.5, color=self.grid_color, linewidth=0.5, alpha=0.5)
        
        # Draw dots (grid points)
        for i in range(rows):
            for j in range(cols):
                ax.plot(j, rows-1-i, 'o', color=self.dot_color, markersize=4, alpha=0.6)
        
        # Draw pattern connections
        self._draw_pattern_connections(ax, grid)
        
        # Customize the plot
        ax.set_xlim(-1, cols)
        ax.set_ylim(-1, rows)
        ax.set_aspect('equal')
        ax.set_title('Kolam Design Canvas\n(Click coordinates below to toggle points)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row (inverted)')
        
        # Remove ticks for cleaner look
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticklabels(range(cols))
        ax.set_yticklabels(range(rows-1, -1, -1))  # Invert y-axis labels
        
        plt.tight_layout()
        return fig
    
    def create_pattern_display(self, grid, title="Kolam Pattern"):
        """
        Create a clean display of a Kolam pattern.
        
        Args:
            grid (np.array): Pattern grid to display
            title (str): Title for the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        rows, cols = grid.shape
        
        # Draw subtle grid background
        for i in range(rows + 1):
            ax.axhline(y=i-0.5, color=self.grid_color, linewidth=0.3, alpha=0.3)
        for j in range(cols + 1):
            ax.axvline(x=j-0.5, color=self.grid_color, linewidth=0.3, alpha=0.3)
        
        # Draw dots
        for i in range(rows):
            for j in range(cols):
                ax.plot(j, rows-1-i, 'o', color=self.dot_color, markersize=3, alpha=0.4)
        
        # Draw pattern with enhanced styling
        self._draw_pattern_connections(ax, grid, enhanced=True)
        
        # Customize the plot
        ax.set_xlim(-1, cols)
        ax.set_ylim(-1, rows)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Remove axis for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def _draw_pattern_connections(self, ax, grid, enhanced=False):
        """Draw the pattern connections on the grid."""
        rows, cols = grid.shape
        
        # Draw active points
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == 1:
                    # Convert grid coordinates to plot coordinates
                    plot_x = j
                    plot_y = rows - 1 - i
                    
                    if enhanced:
                        ax.plot(plot_x, plot_y, 'o', color=self.line_color, 
                               markersize=8, markeredgewidth=2, 
                               markerfacecolor=self.highlight_color,
                               markeredgecolor=self.line_color)
                    else:
                        ax.plot(plot_x, plot_y, 'o', color=self.highlight_color, 
                               markersize=6)
        
        # Draw connections between adjacent active points
        self._draw_connections(ax, grid, enhanced)
    
    def _draw_connections(self, ax, grid, enhanced=False):
        """Draw connections between adjacent active points."""
        rows, cols = grid.shape
        line_width = 3 if enhanced else 2
        
        # Check horizontal connections
        for i in range(rows):
            for j in range(cols - 1):
                if grid[i, j] == 1 and grid[i, j + 1] == 1:
                    y_plot = rows - 1 - i
                    ax.plot([j, j + 1], [y_plot, y_plot], 
                           color=self.line_color, linewidth=line_width)
        
        # Check vertical connections
        for i in range(rows - 1):
            for j in range(cols):
                if grid[i, j] == 1 and grid[i + 1, j] == 1:
                    x_plot = j
                    ax.plot([x_plot, x_plot], [rows - 1 - i, rows - 2 - i], 
                           color=self.line_color, linewidth=line_width)
        
        # Check diagonal connections (if desired for more complex patterns)
        if enhanced:
            self._draw_diagonal_connections(ax, grid, rows, cols, line_width)
    
    def _draw_diagonal_connections(self, ax, grid, rows, cols, line_width):
        """Draw diagonal connections for enhanced visualization."""
        # Main diagonal connections
        for i in range(rows - 1):
            for j in range(cols - 1):
                if grid[i, j] == 1 and grid[i + 1, j + 1] == 1:
                    # Check if this diagonal makes sense (no crossing active horizontal/vertical)
                    y1, y2 = rows - 1 - i, rows - 2 - i
                    ax.plot([j, j + 1], [y1, y2], 
                           color=self.line_color, linewidth=line_width//2, 
                           alpha=0.7, linestyle='--')
        
        # Anti-diagonal connections
        for i in range(rows - 1):
            for j in range(1, cols):
                if grid[i, j] == 1 and grid[i + 1, j - 1] == 1:
                    y1, y2 = rows - 1 - i, rows - 2 - i
                    ax.plot([j, j - 1], [y1, y2], 
                           color=self.line_color, linewidth=line_width//2, 
                           alpha=0.7, linestyle='--')
    
    def visualize_symmetry_analysis(self, grid, analysis_results):
        """
        Visualize symmetry analysis results.
        
        Args:
            grid (np.array): Original pattern
            analysis_results (dict): Results from symmetry analysis
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Symmetry Analysis Results', fontsize=16, fontweight='bold')
        
        # Original pattern
        self._plot_single_pattern(axes[0, 0], grid, "Original Pattern")
        
        # Reflection symmetries
        reflection_types = analysis_results.get('reflection_symmetry', [])
        
        if 'horizontal' in reflection_types:
            reflected = np.flipud(grid)
            self._plot_single_pattern(axes[0, 1], reflected, "Horizontal Reflection")
        else:
            axes[0, 1].text(0.5, 0.5, 'No Horizontal\nReflection Symmetry', 
                          ha='center', va='center', transform=axes[0, 1].transAxes,
                          fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor="lightgray"))
            axes[0, 1].set_title("Horizontal Reflection")
        
        if 'vertical' in reflection_types:
            reflected = np.fliplr(grid)
            self._plot_single_pattern(axes[1, 0], reflected, "Vertical Reflection")
        else:
            axes[1, 0].text(0.5, 0.5, 'No Vertical\nReflection Symmetry', 
                          ha='center', va='center', transform=axes[1, 0].transAxes,
                          fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor="lightgray"))
            axes[1, 0].set_title("Vertical Reflection")
        
        # Rotational symmetry
        rotation_order = analysis_results.get('rotation_order', 1)
        if rotation_order > 1:
            from scipy.ndimage import rotate
            angle = 360 / rotation_order
            rotated = rotate(grid.astype(float), angle, reshape=False, order=0)
            rotated = (rotated > 0.5).astype(int)
            self._plot_single_pattern(axes[1, 1], rotated, f"Rotated {angle:.0f}°")
        else:
            axes[1, 1].text(0.5, 0.5, 'No Rotational\nSymmetry', 
                          ha='center', va='center', transform=axes[1, 1].transAxes,
                          fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor="lightgray"))
            axes[1, 1].set_title("Rotational Symmetry")
        
        plt.tight_layout()
        return fig
    
    def _plot_single_pattern(self, ax, grid, title):
        """Plot a single pattern on given axes."""
        rows, cols = grid.shape
        
        # Draw grid
        for i in range(rows + 1):
            ax.axhline(y=i-0.5, color=self.grid_color, linewidth=0.3, alpha=0.3)
        for j in range(cols + 1):
            ax.axvline(x=j-0.5, color=self.grid_color, linewidth=0.3, alpha=0.3)
        
        # Draw dots
        for i in range(rows):
            for j in range(cols):
                ax.plot(j, rows-1-i, 'o', color=self.dot_color, markersize=2, alpha=0.4)
        
        # Draw pattern
        self._draw_pattern_connections(ax, grid, enhanced=False)
        
        ax.set_xlim(-1, cols)
        ax.set_ylim(-1, rows)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def create_comparison_plot(self, original_grid, generated_grid, analysis_results=None):
        """
        Create a side-by-side comparison of original and generated patterns.
        
        Args:
            original_grid (np.array): Original pattern
            generated_grid (np.array): Generated pattern
            analysis_results (dict): Optional analysis results
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Pattern Comparison', fontsize=16, fontweight='bold')
        
        self._plot_single_pattern(axes[0], original_grid, "Original Pattern")
        self._plot_single_pattern(axes[1], generated_grid, "Generated Pattern")
        
        # Add analysis information if available
        if analysis_results:
            info_text = f"""
Symmetry Score: {analysis_results.get('symmetry_score', 0):.2f}
Connected Components: {analysis_results.get('connected_components', 0)}
Pattern Density: {analysis_results.get('pattern_density', 0):.2f}
Closed Loops: {analysis_results.get('closed_loops', 0)}
            """.strip()
            
            fig.text(0.02, 0.02, info_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def visualize_pattern_evolution(self, pattern_sequence):
        """
        Visualize the evolution of pattern generation.
        
        Args:
            pattern_sequence (list): List of pattern grids showing evolution
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        n_patterns = len(pattern_sequence)
        cols = min(4, n_patterns)
        rows = (n_patterns + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        fig.suptitle('Pattern Generation Evolution', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = [axes] if n_patterns == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, pattern in enumerate(pattern_sequence):
            self._plot_single_pattern(axes[i], pattern, f"Step {i+1}")
        
        # Hide unused subplots
        for i in range(n_patterns, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_analysis_summary_plot(self, analysis_results):
        """
        Create a visual summary of pattern analysis results.
        
        Args:
            analysis_results (dict): Analysis results to visualize
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Pattern Analysis Summary', fontsize=16, fontweight='bold')
        
        # Symmetry score pie chart
        symmetry_score = analysis_results.get('symmetry_score', 0)
        ax1.pie([symmetry_score, 1 - symmetry_score], 
               labels=['Symmetric', 'Asymmetric'],
               colors=[self.highlight_color, self.grid_color],
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Symmetry Score')
        
        # Properties bar chart
        properties = {
            'Density': analysis_results.get('pattern_density', 0),
            'Components': analysis_results.get('connected_components', 0) / 10,
            'Loops': analysis_results.get('closed_loops', 0) / 5,
            'Motifs': analysis_results.get('motif_count', 0) / 5
        }
        
        bars = ax2.bar(properties.keys(), properties.values(), 
                      color=[self.line_color, self.highlight_color, 
                            self.dot_color, '#FFA500'])
        ax2.set_title('Pattern Properties (Normalized)')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, properties.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Reflection symmetry types
        reflection_types = analysis_results.get('reflection_symmetry', [])
        all_types = ['horizontal', 'vertical', 'diagonal1', 'diagonal2']
        type_values = [1 if rtype in reflection_types else 0 for rtype in all_types]
        
        ax3.bar(all_types, type_values, color=self.highlight_color)
        ax3.set_title('Reflection Symmetry Types')
        ax3.set_ylabel('Present')
        ax3.set_ylim(0, 1.2)
        
        # Rotational symmetry
        rotation_order = analysis_results.get('rotation_order', 1)
        rotation_angle = 360 / rotation_order if rotation_order > 1 else 0
        
        angles = np.linspace(0, 2*np.pi, rotation_order, endpoint=False) if rotation_order > 1 else [0]
        radii = [1] * len(angles)
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        ax4.bar(angles, radii, width=2*np.pi/max(rotation_order, 8), 
               color=self.line_color, alpha=0.7)
        ax4.set_title(f'Rotational Symmetry\n({rotation_order}-fold)', 
                     position=(0.5, 1.1))
        ax4.set_ylim(0, 1.5)
        
        plt.tight_layout()
        return fig
