import streamlit as st
import os
import logging
import pandas as pd
from src.prithvi_model import SatelliteImageAnalyzer

def format_table_data(tanks_data, columns):
    """Helper function to format data for tables"""
    df = pd.DataFrame(tanks_data)
    if len(df) > 0:
        df = df[columns]
        if "volume_barrels" in df.columns:
            df["volume_barrels"] = df["volume_barrels"].map("{:,.2f}".format)
        df.columns = [col.replace("_", " ").title() for col in df.columns]
    return df

def main():
    st.set_page_config(
        page_title="Oil Tank Detection System",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("Oil Tank Detection and Volume Analysis System")
    
    # Initialize model and results in session state
    if 'analyzer' not in st.session_state:
        try:
            st.session_state.analyzer = SatelliteImageAnalyzer()
            st.session_state.results = None
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return
    
    # Image selection
    image_dir = "data/satellite_images"
    try:
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            st.warning("No images found. Please add images to data/satellite_images/")
            return
    except FileNotFoundError:
        st.error("Images directory not found")
        os.makedirs(image_dir, exist_ok=True)
        st.info("Created images directory. Please add satellite images.")
        return

    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Image Selection")
        selected_img = st.selectbox("Select Satellite Image:", images)
        image_path = os.path.join(image_dir, selected_img)
        st.image(image_path, caption="Selected Image", use_container_width=True)

    with col2:
        st.subheader("Analysis Controls")
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image for oil tanks..."):
                st.session_state.results = st.session_state.analyzer.detect_industrial_features(image_path)
                
                if "error" in st.session_state.results:
                    st.error(st.session_state.results["error"])
                else:
                    st.success("Analysis Complete!")
                    
                    # Display summary in a table
                    summary_data = {
                        "Metric": ["Total Tanks", "Total Storage Capacity", "Average Detection Confidence"],
                        "Value": [
                            st.session_state.results["metrics"]["total_tanks"],
                            f"{st.session_state.results['metrics']['total_capacity_barrels']:,.2f} barrels",
                            st.session_state.results["metrics"]["confidence_score"]
                        ]
                    }
                    st.subheader("Analysis Summary")
                    st.table(pd.DataFrame(summary_data))
                    
                    # Display detailed results if tanks were detected
                    if st.session_state.results.get("tanks_data"):
                        st.subheader("Detailed Tank Analysis")
                        columns = ["tank_number", "diameter_meters", "volume_barrels", "confidence"]
                        df = format_table_data(st.session_state.results["tanks_data"], columns)
                        st.table(df)
                    
                    # Show visualization
                    vis_path = st.session_state.analyzer.visualize_detections(image_path)
                    if vis_path and os.path.exists(vis_path):
                        st.subheader("Visualization")
                        st.image(vis_path, caption="Detected Tanks with Measurements", use_container_width=True)

        # Q&A Section
        st.subheader("Ask About the Tanks")
        
        with st.expander("Example Questions"):
            st.write("You can ask questions like:")
            st.write("- How many tanks are detected?")
            st.write("- What is the total storage capacity?")
            st.write("- Show me the volume of each tank")
            st.write("- What is the detection confidence?")
            st.write("- Where are the tanks located?")
        
        question = st.text_input("Enter your question:")
        if question and st.button("Get Answer"):
            if st.session_state.results is None:
                st.warning("Please run the analysis first")
            elif "tanks_data" in st.session_state.results:
                st.subheader("Answer")
                
                if "how many" in question.lower():
                    summary_df = pd.DataFrame({
                        "Metric": ["Total Tanks", "Total Capacity", "Average Confidence"],
                        "Value": [
                            st.session_state.results["metrics"]["total_tanks"],
                            f"{st.session_state.results['metrics']['total_capacity_barrels']:,.2f} barrels",
                            st.session_state.results["metrics"]["confidence_score"]
                        ]
                    })
                    st.table(summary_df)
                
                elif "volume" in question.lower() or "capacity" in question.lower():
                    df = format_table_data(
                        st.session_state.results["tanks_data"],
                        ["tank_number", "volume_barrels", "confidence"]
                    )
                    st.table(df)
                
                elif "size" in question.lower() or "diameter" in question.lower():
                    df = format_table_data(
                        st.session_state.results["tanks_data"],
                        ["tank_number", "diameter_pixels", "diameter_meters", "confidence"]
                    )
                    st.table(df)
                
                elif "location" in question.lower() or "where" in question.lower():
                    df = format_table_data(
                        st.session_state.results["tanks_data"],
                        ["tank_number", "position", "confidence"]
                    )
                    st.table(df)
                
                elif "confidence" in question.lower():
                    df = format_table_data(
                        st.session_state.results["tanks_data"],
                        ["tank_number", "confidence"]
                    )
                    st.table(df)
                
                else:
                    # Show complete information
                    df = format_table_data(
                        st.session_state.results["tanks_data"],
                        ["tank_number", "diameter_meters", "volume_barrels", 
                         "position", "confidence"]
                    )
                    st.table(df)
            else:
                st.warning("No tank data available. Please run the analysis first.")

        # Advanced Settings
        with st.expander("Advanced Settings"):
            st.write("Adjust Analysis Parameters")
            new_ratio = st.slider(
                "Height to Diameter Ratio",
                min_value=0.2,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Standard ratio is 0.6 for typical oil storage tanks"
            )
            if new_ratio != st.session_state.analyzer.height_to_diameter_ratio:
                st.session_state.analyzer.height_to_diameter_ratio = new_ratio
                st.info("Ratio updated. Run analysis again to apply changes.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()