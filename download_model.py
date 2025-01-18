import os
from transformers import AutoModelForVisionEncoderDecoder, AutoTokenizer, AutoImageProcessor
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def download_model():
    model_name = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
    logging.info(f"Starting download of {model_name}")
    
    try:
        # Set cache directory explicitly
        cache_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(cache_dir, exist_ok=True)
        logging.info(f"Using cache directory: {cache_dir}")

        logging.info("Downloading model...")
        model = AutoModelForVisionEncoderDecoder.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=False
        )
        logging.info("Model downloaded successfully!")

        logging.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        logging.info("Tokenizer downloaded successfully!")

        logging.info("Downloading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        logging.info("Image processor downloaded successfully!")

        # Save all components
        logging.info("Saving model components...")
        model.save_pretrained(os.path.join(cache_dir, "model"))
        tokenizer.save_pretrained(os.path.join(cache_dir, "tokenizer"))
        image_processor.save_pretrained(os.path.join(cache_dir, "image_processor"))
        
        logging.info("All components downloaded and saved successfully!")
        
    except Exception as e:
        logging.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logging()
    try:
        download_model()
    except KeyboardInterrupt:
        logging.info("Download interrupted by user")
    except Exception as e:
        logging.error(f"Download failed: {str(e)}")