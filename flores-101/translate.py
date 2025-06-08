from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from tqdm import tqdm
import logging
from sacrebleu import corpus_bleu
import os
import json
from sacrebleu.metrics import BLEU, CHRF
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS (Metal Performance Shaders) for acceleration")
else:
    device = torch.device("cpu")
    logger.info("MPS not available, using CPU")

def load_and_prepare_data(target_lang, split="dev"):
    """Load and prepare the FLORES-101 dataset for the specified language."""
    try:
        dataset_name = "facebook/flores"
        source_lang = "eng_Latn"
        
        # Map language codes to their full format with script
        lang_code_map = {
            "hau": "hau_Latn",
            "nso": "nso_Latn",
            "zul": "zul_Latn"
        }
        
        target_lang_full = lang_code_map.get(target_lang)
        if not target_lang_full:
            raise ValueError(f"Unsupported target language code: {target_lang}")
        
        # Load source and target datasets
        source_dataset = load_dataset(dataset_name, name=source_lang, split=split, trust_remote_code=True)
        target_dataset = load_dataset(dataset_name, name=target_lang_full, split=split, trust_remote_code=True)
        
        # Create training pairs
        training_data = {
            "input_text": [src["sentence"] for src in source_dataset],
            "target_text": [tgt["sentence"] for tgt in target_dataset]
        }
        
        # Convert to HuggingFace Dataset
        from datasets import Dataset
        return Dataset.from_dict(training_data)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def initialize_model(model_path="facebook/nllb-200-distilled-600M"):
    """Initialize the NLLB model and tokenizer."""
    try:
        # Check for MPS (Metal Performance Shaders) availability
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) for acceleration")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using {device} for acceleration")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

def translate_english_to_target_lang(model, tokenizer, device, ref_sentences, output_file, target_lang_code, batch_size=16):
    """Translate English sentences to target language using the trained NLLB model."""
    try:
        # Map language codes to NLLB language codes
        nllb_lang_codes = {
            "hau": "hau_Latn",
            "nso": "nso_Latn",
            "zul": "zul_Latn"
        }
        
        target_lang = nllb_lang_codes.get(target_lang_code)
        if not target_lang:
            raise ValueError(f"Unsupported target language code: {target_lang_code}")
        
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        
        logger.info(f"Starting translation to {target_lang}...")
        
        # Prepare all sentences for batch processing
        all_sentences = ref_sentences["input_text"]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            # Process in batches
            for i in tqdm(range(0, len(all_sentences), batch_size), desc=f"Translating to {target_lang}"):
                batch_sentences = all_sentences[i:i + batch_size]
                
                # Tokenize batch without length limits
                inputs = tokenizer(
                    batch_sentences,
                    return_tensors="pt",
                    padding=True,
                    truncation=False
                ).to(device)

                with torch.no_grad():
                    translated = model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        num_beams=4,
                        early_stopping=True
                    )
                
                # Decode and write batch results
                decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
                for translation in decoded:
                    f.write(translation.strip() + "\n")
        
        logger.info(f"Translation completed. Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        raise

def calculate_metrics(hypotheses, references, lang):
    """Calculate various translation metrics."""
    metrics = {}
    
    try:
        # BLEU score
        bleu = BLEU()
        metrics['BLEU'] = bleu.corpus_score(hypotheses, [references]).score
        
        # chrF score
        chrf = CHRF()
        metrics['chrF'] = chrf.corpus_score(hypotheses, [references]).score
        
        # BERTScore
        P, R, F1 = bert_score(hypotheses, references, lang=lang, device='mps' if torch.backends.mps.is_available() else 'cpu')
        metrics['BERTScore'] = F1.mean().item()
        
        # Semantic Similarity Score (replacing COMET)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        hyp_embeddings = model.encode(hypotheses, convert_to_tensor=True)
        ref_embeddings = model.encode(references, convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(hyp_embeddings, ref_embeddings)
        metrics['Semantic_Score'] = similarity.mean().item()
        
        # Error Type Frequency Distribution
        error_types = analyze_errors(hypotheses, references)
        metrics['Error_Distribution'] = error_types
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics

def analyze_errors(hypotheses, references):
    """Analyze translation errors and their distribution."""
    error_types = Counter()
    
    for hyp, ref in zip(hypotheses, references):
        # Word order errors
        hyp_words = set(hyp.split())
        ref_words = set(ref.split())
        if hyp_words == ref_words and hyp != ref:
            error_types['word_order'] += 1
        
        # Missing words
        missing = ref_words - hyp_words
        if missing:
            error_types['missing_words'] += len(missing)
        
        # Extra words
        extra = hyp_words - ref_words
        if extra:
            error_types['extra_words'] += len(extra)
        
        # Case errors
        if hyp.lower() == ref.lower() and hyp != ref:
            error_types['case_errors'] += 1
        
        # Punctuation errors
        hyp_no_punct = re.sub(r'[^\w\s]', '', hyp)
        ref_no_punct = re.sub(r'[^\w\s]', '', ref)
        if hyp_no_punct == ref_no_punct and hyp != ref:
            error_types['punctuation_errors'] += 1
    
    return dict(error_types)

def main():
    # Configuration
    languages = {
        "hausa": "hau",
        "northern-sotho": "nso",
        "zulu": "zul"
    }

    model_name = "nllb-200-distilled-600M"
    model_path = "facebook/nllb-200-distilled-600M"
    output_base_dir = "output"

    # Initialize model and tokenizer
    tokenizer, model, device = initialize_model(model_path)
    
    # Translate for each language
    for lang, code in languages.items():
        output_file = f"{output_base_dir}/{model_name}/flores101.{lang}.hyp.txt"
        
        # Check if translation already exists
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"Translation for {lang} already exists. Skipping translation...")
        else:
            logger.info(f"Translating to {lang}...")
            # Use devtest split for testing
            test_data = load_and_prepare_data(target_lang=code, split="devtest")
            translate_english_to_target_lang(model, tokenizer, device, test_data, output_file, code)
        
        # Calculate metrics
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                hyps = [line.strip() for line in f]

            with open(f"{lang}/flores101.{lang}.ref.test.txt", "r", encoding="utf-8") as f:
                refs = [line.strip() for line in f]

            metrics = calculate_metrics(hyps, refs, lang)
            
            print(f"\nMetrics for {lang}:")
            print("-" * 30)
            for metric_name, value in metrics.items():
                if metric_name != 'Error_Distribution':
                    print(f"{metric_name:15}: {value:.4f}")
            
            if 'Error_Distribution' in metrics:
                print("\nError Distribution:")
                for error_type, count in metrics['Error_Distribution'].items():
                    print(f"{error_type:20}: {count}")
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {lang}: {e}")
            print(f"Could not calculate metrics for {lang}")

    # Print summary of all metrics
    print("\nSummary of all metrics:")
    print("-" * 60)
    print(f"{'Language':15} {'BLEU':>8} {'chrF':>8} {'BERTScore':>10} {'Semantic':>8}")
    print("-" * 60)
    
    for lang in languages:
        try:
            output_file = f"{output_base_dir}/{model_name}/flores101.{lang}.hyp.txt"
            with open(output_file, "r", encoding="utf-8") as f:
                hyps = [line.strip() for line in f]

            with open(f"{lang}/flores101.{lang}.ref.test.txt", "r", encoding="utf-8") as f:
                refs = [line.strip() for line in f]

            metrics = calculate_metrics(hyps, refs, lang)
            print(f"{lang:15} {metrics['BLEU']:8.2f} {metrics['chrF']:8.2f} {metrics['BERTScore']:10.2f} {metrics['Semantic_Score']:8.2f}")
        except Exception as e:
            logger.error(f"Error calculating metrics for {lang}: {e}")
            print(f"{lang:15} {'Error':>8} {'Error':>8} {'Error':>10} {'Error':>8}")
    
    print("-" * 60)

if __name__ == "__main__":
    main()