import torch
import logging
from typing import Dict, List, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QRMRewardModel:
    """
    Implementation of the QRM-Llama3.1-8B-v2 reward model
    """

    def __init__(
        self,
        model_id: str = "nicolinho/QRM-Llama3.1-8B-v2",
        device: Optional[str] = None,
        use_torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Set dtype based on device - this is critical for preventing type mismatches
        if use_torch_dtype is None:
            # Force float32 for all devices to ensure compatibility
            # BFloat16 is causing issues with mixed precision operations
            self.use_torch_dtype = torch.float32
            logger.info(f"Using torch.float32 for {self.device} to ensure compatibility")
        else:
            self.use_torch_dtype = use_torch_dtype
        
        self.cache_dir = cache_dir

        # Load the model and tokenizer
        logger.info(f"Loading QRM Reward model: {model_id} on {self.device}")
        try:
            # Load model with optimal settings for the device
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                torch_dtype=self.use_torch_dtype,  # Force specified dtype
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            
            # Cast model parameters to float32 to ensure compatibility
            self.model = self.model.to(dtype=torch.float32)
            
            # Move model to device after ensuring dtype
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Check for NVIDIA GPUs and apply optimizations
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                logger.info(f"Running on GPU: {gpu_name}")
                
                # Check specifically for GH200 or other Hopper architecture GPUs
                if "gh200" in gpu_name or "hopper" in gpu_name or "h100" in gpu_name:
                    logger.info("Detected Grace Hopper architecture")
                    # No autocast or mixed precision for now - using float32 for compatibility
                elif "a100" in gpu_name:
                    logger.info("Detected A100 GPU")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                cache_dir=cache_dir
            )
            logger.info("QRM Reward model loaded successfully")

            # Get reward attributes
            self.attributes = ['helpsteer-helpfulness', 'helpsteer-correctness',
                          'helpsteer-coherence', 'helpsteer-complexity',
                          'helpsteer-verbosity']

        except Exception as e:
            logger.error(f"Failed to load QRM Reward model: {str(e)}")
            raise

    def format_prompt(self, prompt: str, response: str) -> List[Dict[str, str]]:
        """Format prompt and response into the chat template format"""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return messages

    def to(self, device):
        """Move the model to a different device properly"""
        logger.info(f"Moving QRM reward model from {self.device} to {device}")
        self.device = device

        # Always keep the model in float32 for all devices
        # This avoids dtype issues with mixed precision operations
        if hasattr(self, 'model'):
            self.model = self.model.to(dtype=torch.float32, device=device)

        return self

    def get_reward_score(self, prompt: str, response: str) -> float:
        """Get reward score for a prompt-response pair"""
        try:
            # Format into messages
            messages = self.format_prompt(prompt, response)

            # Apply tokenization
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            )

            # Move inputs to model's device with explicit float32 cast for inputs
            input_ids = input_ids.to(self.device)

            # Get reward prediction without autocast to avoid dtype mismatches
            with torch.no_grad():
                output = self.model(input_ids)
                # Convert to CPU float32 for consistency
                reward = output.score.detach().cpu().float().item()

            return reward

        except Exception as e:
            logger.error(f"Error getting reward score: {str(e)}")
            # Add more debug info
            logger.error(f"Device: {self.device}, Model dtype: {next(self.model.parameters()).dtype}")
            # Return a default value on error
            return 0.0
