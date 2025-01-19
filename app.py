import streamlit as st
import os
import logging
import pandas as pd
import cv2
from src.prithvi_model import SatelliteImageAnalyzer

def format_large_number(value):
    """Format large numbers into millions (M) or billions (B) with green color"""
    billion = 1_000_000_000
    million = 1_000_000
    
    if value >= billion:
        amount = f"${value / billion:.2f}B"
    elif value >= million:
        amount = f"${value / million:.2f}M"
    else:
        amount = f"${value:,.0f}"
    
    return f'<span style="color: #00c853">{amount}</span>'

def generate_response(question, results):
    """Generate response based on question and analysis results"""
    try:
        image_path = results.get("image_path")
        if not image_path:
            return "Please run the analysis first."
            
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return "Could not load the image for analysis."
        
        # Check if it's a tank-specific question
        if any(word in question.lower() for word in ["tank", "barrel", "volume", "capacity"]):
            return generate_tank_response(question, results)
        
        # Generate real-time analysis for other questions
        response = st.session_state.analyzer.analyze_image_realtime(image, question)
        if response:
            return f"""<div class="analysis-response">
                <h4>Analysis Results</h4>
                <p>{response}</p>
            </div>"""
        else:
            return "I couldn't analyze that aspect of the image. Please try another question."
            
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return "I encountered an error while processing your question. Please try again."

def generate_tank_response(question, results):
    """Generate response for tank-specific questions"""
    if "tanks_data" not in results:
        return "No tank data available. Please run the analysis first."
        
    if "how many" in question.lower():
        total_volume = results["metrics"]["total_capacity_barrels"]
        total_oil_value = total_volume * 80.73
        summary_df = pd.DataFrame({
            "Metric": ["Total Tanks", "Total Capacity", "Total Oil Value", "Average Confidence"],
            "Value": [
                results["metrics"]["total_tanks"],
                f"{total_volume:,.0f} barrels",
                format_large_number(total_oil_value),
                results["metrics"]["confidence_score"]
            ]
        })
        return summary_df.to_html(escape=False, index=False)
    
    # Return detailed tank information
    df = format_table_data(
        results["tanks_data"],
        ["tank_number", "diameter_meters", "volume_barrels", "oil_value_usd", "confidence"]
    )
    return df.to_html(escape=False, index=False)

def format_table_data(tanks_data, columns):
    """Helper function to format data for tables"""
    df = pd.DataFrame(tanks_data)
    if len(df) > 0:
        df = df[columns]
        if "volume_barrels" in df.columns:
            df["volume_barrels"] = df["volume_barrels"].map("{:,.0f}".format)
        if "oil_value_usd" in df.columns:
            df["oil_value_usd"] = df["oil_value_usd"].map(format_large_number)
        df.columns = [col.replace("_", " ").title() for col in df.columns]
    return df

def main():
    st.set_page_config(
        page_title="Earth Analysis",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("Earth Analysis")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SatelliteImageAnalyzer()
        st.session_state.results = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Image selection and analysis section
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Image Selection")
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

        selected_img = st.selectbox("Select Satellite Image:", images)
        image_path = os.path.join(image_dir, selected_img)
        st.image(image_path, caption="Selected Image", use_container_width=True)

    with col2:
        st.subheader("Analysis Controls")
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                st.session_state.results = st.session_state.analyzer.detect_industrial_features(image_path)
                st.session_state.messages = []  # Clear chat history
                
                if "error" in st.session_state.results:
                    st.error(st.session_state.results["error"])
                else:
                    st.success("Analysis Complete!")
                    
                    # Calculate total oil value
                    total_volume = st.session_state.results["metrics"]["total_capacity_barrels"]
                    total_oil_value = total_volume * 80.73
                    
                    # Display summary
                    summary_data = {
                        "Metric": [
                            "Total Tanks", 
                            "Total Storage Capacity", 
                            "Total Oil Value ($80.73/bbl)", 
                            "Average Detection Confidence"
                        ],
                        "Value": [
                            st.session_state.results["metrics"]["total_tanks"],
                            f"{total_volume:,.0f} barrels",
                            format_large_number(total_oil_value),
                            st.session_state.results["metrics"]["confidence_score"]
                        ]
                    }
                    st.subheader("Analysis Summary")
                    st.markdown(pd.DataFrame(summary_data).to_html(escape=False, index=False), 
                              unsafe_allow_html=True)
                    
                    # Show visualization
                    vis_path = st.session_state.analyzer.visualize_detections(image_path)
                    if vis_path and os.path.exists(vis_path):
                        st.subheader("Visualization")
                        st.image(vis_path, caption="Detected Features", 
                               use_container_width=True)

    # Chat interface
    st.divider()
    st.subheader("Chat Analysis")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("is_html", False):
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the image..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if st.session_state.results is None:
                response = "Please run the analysis first."
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = generate_response(prompt, st.session_state.results)
                st.markdown(response, unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "is_html": True
                })

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