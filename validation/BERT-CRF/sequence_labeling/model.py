import torch
from torch import nn
import torch.functional as F

class LangModelWithDense(nn.Module):
    def __init__(self, lang_model, emb_size, num_classes, fine_tune):
        """
        Create the model.

        :param lang_model: The language model.
        :param emb_size: The size of the contextualized embeddings.
        :param num_classes: The number of classes.
        :param fine_tune: whether to fine-tune or freeze the language model's weights.
        """
        super().__init__()
        self.num_classes = num_classes
        self.fine_tune = fine_tune

        self.lang_model = lang_model

        self.linear = nn.Linear(emb_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        """
        Forward function of the model.

        :param x: The inputs. Shape: [batch_size, seq_len].
        :param mask: The attention mask. Ones are unmasked, zeros are masked.
            Shape: [batch_size, seq_len].
        :return: The logits. Shape: [batch_size, seq_len, num_classes].

        Example:
        model = LangModelWithDense(...)

        x = np.array([[2, 2], [1, 3]])
        mask = np.array([[1, 1], [1, 0]])

        logits = model.foward(x, mask)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # this will modify the language model's weights
        if not self.fine_tune:
            with torch.no_grad():
                self.lang_model.eval()
                embeddings = self.lang_model(x, attention_mask=mask)[0]
        # this will not
        else:
            embeddings = self.lang_model(x, attention_mask=mask)[0]

        # create a vector to retain the output for each token. Shape:
        # [batch_size, seq_len, num_classes]
        logits = torch.zeros((batch_size, seq_len, self.num_classes))

        # feed-forward for each token in the sequence and save it in outputs
        for i in range(seq_len):
            # the logits for a single token. Shape: [batch_size, num_classes]
            logit = self.dropout(self.linear(embeddings[:, i, :]))

            logits[:, i, :] = logit

        return logits
    
    def predict_with_confidence(self, x, mask, confidence_method='average'):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, mask)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Get the highest probability for each token in the sequence
            max_probs, predictions = torch.max(probs, dim=-1)

            # Calculate overall sequence confidence based on selected method
            if confidence_method == 'average':
                # Average of the maximum probabilities for all tokens in the sequence
                seq_confidences = max_probs.mean(dim=-1)
            elif confidence_method == 'product':
                # Product of the maximum probabilities for all tokens in the sequence
                seq_confidences = max_probs.prod(dim=-1)
            else:
                raise ValueError("Unsupported confidence_method. Choose from ['average', 'product']")

            # Convert tensor to list of confidence scores and apply percentage conversion
            confidence_scores = [round(float(seq_conf)*100, 2) for seq_conf in seq_confidences]
        
        self.train()
        return predictions, confidence_scores
