from pydoc import text
import torch
import torch.nn as nn

from transformers import BertModel

# Re-using the TinyBERT class from the previous iteration, configured as a feature extractor.
class SmallBERT(nn.Module):
    """
    A custom SmallBERT model class configured as a feature extractor.
    Returns the last hidden states for further processing.
    """
    def __init__(self):
        super(SmallBERT, self).__init__()
        # Load the pre-trained SmallBERT model.
        self.smallbert = BertModel.from_pretrained('prajjwal1/bert-small')
        for param in self.smallbert.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Performs the forward pass of the SmallBERT model.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
                                      Shape: (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Tensor indicating which tokens
                                                     should be attended to (1 for real tokens,
                                                     0 for padding tokens).
                                                     Shape: (batch_size, sequence_length).
                                                     Defaults to None.

        Returns:
            torch.Tensor: The last hidden states from the SmallBERT model.
                          Shape: (batch_size, sequence_length, hidden_size).
        """
        output = self.smallbert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state
    
def build_bert(args):
    """
    Builds the SmallBERT model for feature extraction.

    Args:
        args: Arguments containing model configuration.

    Returns:
        SmallBERT: An instance of the SmallBERT model.
    """
    model = SmallBERT()
    model.output_dim = 512 # model.smallbert.config.hidden_size
    model.max_text_len = args.max_text_len if hasattr(args, 'max_text_len') else 16  # Default to 16 if not specified
    return model