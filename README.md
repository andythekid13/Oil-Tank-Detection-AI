# Earth Analysis: Oil Tank Detection System

## Overview
Earth Analysis is a sophisticated tool designed for analyzing satellite imagery to detect and measure oil storage tanks. The system provides detailed analysis including tank dimensions, storage capacity, and estimated oil value based on current market prices.

## Features
- **Automated Tank Detection**: Uses computer vision to identify circular storage tanks in satellite imagery
- **Volume Calculation**: Estimates tank capacity based on detected dimensions
- **Financial Analysis**: Calculates potential oil value based on current market prices
- **Interactive Chat Interface**: Natural language interaction for querying analysis results
- **Real-time Visualization**: Visual overlay of detected tanks with measurements and values

## Key Capabilities
- Tank diameter measurement in meters
- Volume calculation in barrels
- Oil value estimation in USD
- Confidence scoring for detections
- Interactive analysis through chat interface

## Technical Components

### Core Technologies
- Python 3.8+
- OpenCV for image processing
- Streamlit for web interface
- Pandas for data handling

### Key Files
- `app.py`: Main application and UI
- `src/prithvi_model.py`: Core analysis engine
- `requirements.txt`: Project dependencies

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd earth-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare the data directory:
```bash
mkdir -p data/satellite_images
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Image Selection**
   - Upload or select satellite images from the data/satellite_images directory
   - Supported formats: JPG, PNG, JPEG

2. **Analysis**
   - Click "Analyze Image" to detect tanks
   - View summary statistics and detailed measurements
   - Examine visual overlay of detected tanks

3. **Chat Interface**
   - Ask questions about the analysis
   - Query specific tank information
   - Get detailed measurements and values

## Data Analysis Features

### Tank Detection
- Circular object detection using Hough Circles
- Size filtering based on typical tank dimensions
- Confidence scoring based on image features

### Measurements
- Pixel to meter conversion
- Volume calculation using standard tank ratios
- Financial value estimation

### Visualization
- Green circles indicating detected tanks
- Text overlays showing:
  - Volume in barrels
  - Estimated value in USD
  - Detection confidence

## System Requirements
- Python 3.8 or higher
- 4GB RAM minimum
- Support for OpenCV operations
- Modern web browser for UI

## Limitations
- Best suited for high-resolution satellite imagery
- Assumes standard cylindrical tank shapes
- Fixed oil price (currently set to $80.73/barrel)

## Future Enhancements
- Dynamic oil price integration
- Support for different tank shapes
- Enhanced terrain analysis
- Multi-image batch processing
- Historical analysis capabilities

## Contributing
Contributions are welcome! Please read our contributing guidelines and submit pull requests for any enhancements.

## License
[Specify your license here]

## Support
For support and questions, please open an issue in the repository or contact [your contact information].

## Acknowledgments
- OpenCV for computer vision capabilities
- Streamlit for the web interface
- Contributors and maintainers