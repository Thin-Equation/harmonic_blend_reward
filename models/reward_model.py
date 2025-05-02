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
        from utils.device_utils import get_torch_dtype_for_device

        self.model_id = model_id

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Set appropriate dtype for the device if not specified
        if use_torch_dtype is None:
            # For CPU, always use float32 to avoid compatibility issues
            if self.device == "cpu":
                self.use_torch_dtype = torch.float32
                logger.info("Using float32 for CPU to ensure compatibility")
            else:
                self.use_torch_dtype = get_torch_dtype_for_device(torch.device(self.device))
        else:
            self.use_torch_dtype = use_torch_dtype
            
        logger.info(f"Using dtype: {self.use_torch_dtype} for device: {self.device}")
        self.cache_dir = cache_dir

        # Load the model and tokenizer
        logger.info(f"Loading QRM Reward model: {model_id} on {self.device}")
        try:
            # Load model with optimal settings for the device
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                torch_dtype=self.use_torch_dtype,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            # Move model to device after loading
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Check for CUDA and optimize if available
            if self.device == "cuda" and torch.cuda.is_available():
                # Apply optimizations for A100
                gpu_name = torch.cuda.get_device_name(0).lower()
                if "a100" in gpu_name:
                    logger.info("Applying optimizations for A100 GPU")
                    # Flash attention optimization if available
                    if hasattr(self.model.config, "use_flash_attention_2"):
                        self.model.config.use_flash_attention_2 = True
                        logger.info("Enabled Flash Attention 2")
            
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

        # Update dtype if needed when changing devices
        if device == "cpu" and self.use_torch_dtype != torch.float32:
            logger.info("Switching to float32 for CPU compatibility")
            self.use_torch_dtype = torch.float32

        # Move the model to the specified device
        if hasattr(self, 'model'):
            self.model = self.model.to(device)

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

            # Ensure input is on the right device
            input_ids = input_ids.to(self.device)

            # Get reward prediction with proper type handling
            with torch.no_grad():
                # Avoid using autocast for CPU
                if self.device == "cpu":
                    output = self.model(input_ids)
                else:
                    # Use autocast for GPU
                    with torch.autocast(device_type=self.device_type(), dtype=self.use_torch_dtype):
                        output = self.model(input_ids)
                
                # Make sure to fetch from the same device
                reward = output.score.detach().cpu().float().item()

            return reward

        except Exception as e:
            logger.error(f"Error getting reward score: {str(e)}")
            # Return a default value on error
            return 0.0
            
    def device_type(self):
        """Get device type string for autocast"""
        if self.device == "cuda":
            return "cuda"
        elif self.device == "mps":
            return "mps"
        else:
            return "cpu"
