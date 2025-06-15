from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from tqdm import tqdm
import logging
from sacrebleu import corpus_bleu
import os
import json

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

def initialize_model(model_name="facebook/nllb-200-distilled-600M"):
    """Initialize and return the translation model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Use MPS backend for Apple Silicon
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        model = model.to(device)
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

def train_model(model, tokenizer, training_data, output_dir, target_lang_code):
    """Train the NLLB model on the prepared data."""
    try:
        logger.info("Starting model training...")
        
        # Map language codes to NLLB language codes
        nllb_lang_codes = {
            "hau": "hau_Latn",
            "nso": "nso_Latn",
            "zul": "zul_Latn"
        }
        
        target_lang = nllb_lang_codes.get(target_lang_code)
        if not target_lang:
            raise ValueError(f"Unsupported target language code: {target_lang_code}")
        
        # Set target language for tokenizer
        tokenizer.tgt_lang = target_lang
        
        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()
        
        # Prepare dataset
        def preprocess_function(examples):
            # Tokenize inputs with shorter max length
            model_inputs = tokenizer(
                examples["input_text"],
                max_length=64,  # Reduced from 128
                truncation=True,
                padding="max_length"
            )
            
            # Tokenize targets with shorter max length
            labels = tokenizer(
                text_target=examples["target_text"],
                max_length=64,  # Reduced from 128
                truncation=True,
                padding="max_length"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Process the dataset
        processed_dataset = training_data.map(
            preprocess_function,
            batched=True,
            remove_columns=training_data.column_names
        )
        
        # Create training arguments with reduced memory usage
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,  # Reduced from 8
            per_device_eval_batch_size=4,   # Reduced from 8
            learning_rate=2e-5,
            weight_decay=0.01,
            save_total_limit=3,
            save_steps=500,
            logging_steps=100,
            warmup_steps=500,
            push_to_hub=False,
            predict_with_generate=True,
            fp16=False,
            gradient_accumulation_steps=8,  # Increased from 4
            gradient_checkpointing=True,    # Enable gradient checkpointing
            optim="adamw_torch",           # Use PyTorch's AdamW optimizer
            max_grad_norm=1.0,             # Gradient clipping
            dataloader_num_workers=0,      # Disable multiprocessing for dataloader
            dataloader_pin_memory=False    # Disable pin memory
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            eval_dataset=processed_dataset.select(range(50)),  # Reduced from 100
            tokenizer=tokenizer
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def translate_english_to_target_lang(model, tokenizer, device, ref_sentences, output_file, target_lang_code, batch_size=16):
    """Translate English sentences to target language using the trained NLLB model."""
    try:
        tokenizer, model = initialize_model()
        
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
    
    # Train and translate for each language
    for lang, code in languages.items():
        logger.info(f"Training model for {lang}...")
        # Use dev split for training
        training_data = load_and_prepare_data(target_lang=code, split="dev")
        model, tokenizer = train_model(model, tokenizer, training_data, output_dir=f"{output_base_dir}/{model_name}/{lang}", target_lang_code=code)
        
        # Use devtest split for testing
        logger.info(f"Finished training {lang}... proceeding to translate")
        test_data = load_and_prepare_data(target_lang=code, split="devtest")
        output_file = f"{output_base_dir}/{model_name}/flores101.{lang}.hyp.txt"
        translate_english_to_target_lang(model, tokenizer, device, test_data, output_file, code)
        
        # Calculate BLEU score
        with open(output_file, "r", encoding="utf-8") as f:
            hyps = [line.strip() for line in f]

        with open(f"{lang}/flores101.{lang}.ref.test.txt", "r", encoding="utf-8") as f:
            refs = [[line.strip() for line in f]]

        bleu_score = corpus_bleu(hyps, refs)
        print(f"BLEU score for {lang}: {bleu_score.score:.2f}")

    # Print summary of all BLEU scores
    print("\nSummary of BLEU scores:")
    print("-" * 30)
    for lang in languages:
        with open(f"{output_base_dir}/{model_name}/flores101.{lang}.hyp.txt", "r", encoding="utf-8") as f:
            hyps = [line.strip() for line in f]

        with open(f"{lang}/flores101.{lang}.ref.test.txt", "r", encoding="utf-8") as f:
            refs = [[line.strip() for line in f]]

        bleu_score = corpus_bleu(hyps, refs)
        print(f"{lang:15} : {bleu_score.score:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    main()