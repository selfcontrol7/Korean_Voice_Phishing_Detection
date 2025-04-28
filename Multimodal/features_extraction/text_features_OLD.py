import json
import os
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

# Initialize models once
# KoBERT model
_kobert_tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
_kobert_model = BertModel.from_pretrained('monologg/kobert').eval()

# SR-SBERT model (Korean Sentence-BERT)
_sbert_model = SentenceTransformer('jhgan/ko-sbert-nli')

def load_transcript(transcript_path):
    """
    Load transcript from JSON file and extract text
    
    Args:
        transcript_path (str): Path to transcript JSON file
        
    Returns:
        str: Full text of the transcript
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Extract full text from transcript
    if 'text' in transcript_data:
        return transcript_data['text']
    
    # If full text is not available, concatenate segment texts
    if 'segments' in transcript_data:
        return ' '.join([segment['text'] for segment in transcript_data['segments']])
    
    return ""

def extract_kobert_features(text, max_length=512):
    """
    Extract features from text using KoBERT model
    
    Args:
        text (str): Input text
        max_length (int): Maximum sequence length
        
    Returns:
        numpy.ndarray: KoBERT features
    """
    # Tokenize text
    inputs = _kobert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    
    # Extract features
    with torch.no_grad():
        outputs = _kobert_model(**inputs)
    
    # Use the [CLS] token embedding as the sentence representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    
    return cls_embedding

def extract_sbert_features(text):
    """
    Extract features from text using SR-SBERT model
    
    Args:
        text (str): Input text
        
    Returns:
        numpy.ndarray: SR-SBERT features
    """
    # Encode text to get sentence embedding
    embedding = _sbert_model.encode(text, convert_to_numpy=True)
    
    return embedding

def extract_text_features(transcript_path):
    """
    Extract both KoBERT and SR-SBERT features from a transcript file
    
    Args:
        transcript_path (str): Path to transcript JSON file
        
    Returns:
        dict: Dictionary containing KoBERT and SR-SBERT features
    """
    # Load transcript text
    text = load_transcript(transcript_path)
    
    if not text:
        return {
            'kobert': np.zeros(768),  # Default size for BERT embeddings
            'sbert': np.zeros(768)    # Default size for SBERT embeddings
        }
    
    # Extract features
    kobert_features = extract_kobert_features(text)
    sbert_features = extract_sbert_features(text)
    
    return {
        'kobert': kobert_features,
        'sbert': sbert_features
    }

def process_manifest(manifest_path, output_dir=None):
    """
    Process all transcripts in a manifest file and extract features
    
    Args:
        manifest_path (str): Path to manifest file
        output_dir (str, optional): Directory to save features
        
    Returns:
        dict: Dictionary mapping call_ids to features
    """
    features = {}
    
    # Create output directories if needed
    if output_dir:
        os.makedirs(os.path.join(output_dir, 'kobert'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'sbert'), exist_ok=True)
    
    # Read manifest file
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            call_id = entry['call_id']
            transcript_path = entry['transcript_filepath']
            
            # Extract features
            text_features = extract_text_features(transcript_path)
            features[call_id] = text_features
            
            # Save features if output directory is provided
            if output_dir:
                np.save(os.path.join(output_dir, 'kobert', f"{call_id}.npy"), text_features['kobert'])
                np.save(os.path.join(output_dir, 'sbert', f"{call_id}.npy"), text_features['sbert'])
    
    return features