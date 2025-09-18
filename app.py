import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import io
import cv2
from kolam_analyzer import KolamAnalyzer
from pattern_generator import PatternGenerator
from visualization import KolamVisualizer
from utils import grid_to_image, image_to_grid
from sample_kolams import get_sample_kolams

# Initialize session state
if 'grid' not in st.session_state:
    st.session_state.grid = np.zeros((15, 15), dtype=int)
if 'grid_size' not in st.session_state:
    st.session_state.grid_size = 15
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'generated_pattern' not in st.session_state:
    st.session_state.generated_pattern = None

def main():
    st.set_page_config(
        page_title="Kolam Design Analyzer and Generator",
        page_icon="🕸️",
        layout="wide"
    )
    
    st.title("🕸️ Kolam Design Analyzer and Generator")
    st.markdown("*Analyze and generate traditional Indian geometric patterns*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Grid size selector
        new_grid_size = st.slider("Grid Size", min_value=10, max_value=25, value=st.session_state.grid_size)
        if new_grid_size != st.session_state.grid_size:
            st.session_state.grid_size = new_grid_size
            st.session_state.grid = np.zeros((new_grid_size, new_grid_size), dtype=int)
            st.rerun()
        
        # Pattern type selector
        pattern_type = st.selectbox(
            "Pattern Type",
            ["Closed Loop", "Open Path", "Radial", "Linear"]
        )
        
        # Regional style selector
        regional_style = st.selectbox(
            "Regional Style",
            ["Tamil Nadu", "Andhra Pradesh", "Karnataka", "Kerala", "General"]
        )
        
        st.divider()
        
        # Clear grid button
        if st.button("Clear Grid", type="secondary"):
            st.session_state.grid = np.zeros((st.session_state.grid_size, st.session_state.grid_size), dtype=int)
            st.session_state.analysis_results = None
            st.session_state.generated_pattern = None
            st.rerun()
        
        # Load sample patterns
        st.subheader("Sample Patterns")
        sample_kolams = get_sample_kolams()
        selected_sample = st.selectbox("Load Sample", ["None"] + list(sample_kolams.keys()))
        
        if selected_sample != "None" and st.button("Load Sample"):
            st.session_state.grid = sample_kolams[selected_sample]
            st.session_state.analysis_results = None
            st.session_state.generated_pattern = None
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Design Canvas")
        
        # File upload option
        uploaded_file = st.file_uploader("Upload Kolam Image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.grid = image_to_grid(image, st.session_state.grid_size)
            st.session_state.analysis_results = None
            st.session_state.generated_pattern = None
            st.rerun()
        
        # Interactive grid for drawing
        visualizer = KolamVisualizer()
        fig = visualizer.create_interactive_grid(st.session_state.grid)
        
        # Display the grid
        grid_container = st.container()
        with grid_container:
            st.pyplot(fig)
            plt.close(fig)
        
        # Grid interaction buttons
        st.markdown("**Click coordinates to toggle connections:**")
        col_a, col_b = st.columns(2)
        
        with col_a:
            row = st.number_input("Row", min_value=0, max_value=st.session_state.grid_size-1, value=0)
        with col_b:
            col = st.number_input("Column", min_value=0, max_value=st.session_state.grid_size-1, value=0)
        
        if st.button("Toggle Point"):
            st.session_state.grid[row, col] = 1 - st.session_state.grid[row, col]
            st.rerun()
    
    with col2:
        st.subheader("Analysis & Generation")
        
        # Analyze current pattern
        if st.button("Analyze Pattern", type="primary"):
            if np.sum(st.session_state.grid) > 0:
                analyzer = KolamAnalyzer()
                st.session_state.analysis_results = analyzer.analyze_pattern(st.session_state.grid)
                st.rerun()
            else:
                st.warning("Please draw or load a pattern first!")
        
        # Display analysis results
        if st.session_state.analysis_results:
            st.success("Pattern Analysis Complete!")
            
            results = st.session_state.analysis_results
            
            with st.expander("Symmetry Analysis", expanded=True):
                st.write(f"**Reflection Symmetry:** {results['reflection_symmetry']}")
                st.write(f"**Rotational Symmetry:** {results['rotational_symmetry']}° intervals")
                st.write(f"**Symmetry Score:** {results['symmetry_score']:.2f}")
            
            with st.expander("Pattern Properties"):
                st.write(f"**Connected Components:** {results['connected_components']}")
                st.write(f"**Closed Loops:** {results['closed_loops']}")
                st.write(f"**Pattern Density:** {results['pattern_density']:.2f}")
            
            with st.expander("Mathematical Properties"):
                st.write(f"**Euler Characteristic:** {results.get('euler_characteristic', 'N/A')}")
                st.write(f"**Motif Count:** {results.get('motif_count', 0)}")
        
        st.divider()
        
        # Generate new patterns
        st.subheader("Generate New Pattern")
        
        generation_method = st.radio(
            "Generation Method",
            ["Based on Analysis", "Random Symmetric", "Template Based"]
        )
        
        if st.button("Generate Pattern", type="primary"):
            generator = PatternGenerator()
            
            if generation_method == "Based on Analysis" and st.session_state.analysis_results:
                st.session_state.generated_pattern = generator.generate_from_analysis(
                    st.session_state.analysis_results, 
                    st.session_state.grid_size
                )
            elif generation_method == "Random Symmetric":
                symmetry_type = st.selectbox("Symmetry Type", ["reflection", "rotation", "both"])
                st.session_state.generated_pattern = generator.generate_symmetric_pattern(
                    st.session_state.grid_size, 
                    symmetry_type
                )
            else:
                st.session_state.generated_pattern = generator.generate_template_based(
                    st.session_state.grid_size, 
                    pattern_type.lower().replace(" ", "_")
                )
            
            st.rerun()
        
        # Display generated pattern
        if st.session_state.generated_pattern is not None:
            st.success("New pattern generated!")
            
            # Show generated pattern
            fig_gen = visualizer.create_pattern_display(st.session_state.generated_pattern)
            st.pyplot(fig_gen)
            plt.close(fig_gen)
            
            # Option to replace current pattern
            col_x, col_y = st.columns(2)
            with col_x:
                if st.button("Use Generated Pattern"):
                    st.session_state.grid = st.session_state.generated_pattern.copy()
                    st.session_state.analysis_results = None
                    st.rerun()
            
            with col_y:
                # Export functionality
                if st.button("Export as PNG"):
                    img = grid_to_image(st.session_state.generated_pattern)
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Generated Pattern",
                        data=img_buffer,
                        file_name="kolam_pattern.png",
                        mime="image/png"
                    )
    
    # Documentation section
    with st.expander("About Kolam Patterns & Mathematics", expanded=False):
        st.markdown("""
        ### Kolam Patterns
        
        Kolams are traditional floor drawings from South India, created using rice flour or chalk powder. 
        They demonstrate sophisticated mathematical concepts:
        
        **Mathematical Principles:**
        - **Symmetry Groups**: Kolams often exhibit reflection and rotational symmetries
        - **Topology**: Many patterns form closed loops without intersecting lines
        - **Tessellations**: Repeating motifs that tile the plane
        - **Graph Theory**: Patterns can be analyzed as mathematical graphs
        
        **Cultural Significance:**
        - Daily ritual practice in Tamil Nadu, Andhra Pradesh, and Karnataka
        - Represents cosmic order and mathematical beauty
        - Passed down through generations as cultural knowledge
        
        **Design Rules:**
        - Start and end at the same point (closed loops)
        - Lines should not intersect except at dots
        - Symmetric arrangements are preferred
        - Patterns should be drawable in one continuous motion
        """)

if __name__ == "__main__":
    main()
