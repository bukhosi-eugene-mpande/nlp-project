import os
from pathlib import Path
from sacrebleu import corpus_bleu, corpus_chrf
from bert_score import score as bert_score
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def load_nllb_model(model_name="facebook/nllb-200-distilled-600M"):
    print("Loading NLLB model and tokenizer (this may take some time)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def score_sentence_nllb(tokenizer, model, device, src_sentence, hyp_sentence, src_lang="eng_Latn", tgt_lang="hau_Latn"):
    # Set language codes for tokenizer
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    # Tokenize input and target
    inputs = tokenizer(src_sentence, return_tensors="pt", padding=True, truncation=True)
    labels = tokenizer(text_target=hyp_sentence, return_tensors="pt", padding=True, truncation=True).input_ids

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs.loss.item()
    ppl = torch.exp(torch.tensor(loss)).item()
    return loss, ppl

def evaluate_metrics(refs, hyps, srcs, lang_code, comet_model, nllb_tokenizer, nllb_model, nllb_device, dataset_version=""):
    print(f"\n--- {lang_code} ({dataset_version}) ---")

    # BLEU
    bleu = corpus_bleu(hyps, [refs])
    print(f"BLEU: {bleu.score:.2f}")

    # chrF
    chrf = corpus_chrf(hyps, [refs])
    print(f"chrF: {chrf.score:.2f}")

    # BERTScore
    short_lang = lang_code.split('_')[0]
    P, R, F1 = bert_score(hyps, refs, lang=short_lang, rescale_with_baseline=True)
    print(f"BERTScore F1: {F1.mean():.4f}")

    # COMET
    comet_data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)]
    comet_score = comet_model.predict(comet_data, batch_size=8, progress_bar=False)
    print(f"COMET: {comet_score.system_score:.4f}")

    # NLLB scoring
    total_loss = 0
    total_ppl = 0
    for src, hyp in zip(srcs, hyps):
        loss, ppl = score_sentence_nllb(nllb_tokenizer, nllb_model, nllb_device, src, hyp, src_lang="eng_Latn", tgt_lang=lang_code)
        total_loss += loss
        total_ppl += ppl

    avg_loss = total_loss / len(hyps)
    avg_ppl = total_ppl / len(hyps)
    print(f"NLLB Avg Loss: {avg_loss:.4f}")
    print(f"NLLB Avg Perplexity: {avg_ppl:.4f}")

    # Return scores for comparison
    return {
        'bleu': bleu.score,
        'chrf': chrf.score,
        'bertscore_f1': F1.mean().item(),
        'comet': comet_score.system_score,
        'nllb_loss': avg_loss,
        'nllb_ppl': avg_ppl
    }

def compare_datasets(original_scores, corrected_scores, lang_code):
    print(f"\n=== COMPARISON FOR {lang_code} ===")
    print(f"{'Metric':<15} {'Original':<10} {'Corrected':<10} {'Difference':<12} {'% Change':<10}")
    print("-" * 65)
    
    for metric in original_scores.keys():
        orig = original_scores[metric]
        corr = corrected_scores[metric]
        diff = corr - orig
        pct_change = (diff / orig) * 100 if orig != 0 else 0
        
        print(f"{metric:<15} {orig:<10.4f} {corr:<10.4f} {diff:<12.4f} {pct_change:<10.2f}%")

def evaluate_selected_languages_comparison(original_dir, corrected_dir, devtest_dir):
    """
    Compare evaluation on original vs corrected FLORES datasets
    
    Args:
        original_dir: Directory containing original FLORES dev files
        corrected_dir: Directory containing corrected FLORES dev files  
        devtest_dir: Directory containing devtest files (translations to evaluate)
    """
    selected_langs = ['hau_Latn', 'zul_Latn', 'nso_Latn', 'tso_Latn']

    print("Loading COMET model...")
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    nllb_tokenizer, nllb_model, nllb_device = load_nllb_model()

    for lang in selected_langs:
        original_dev_file = Path(original_dir) / f"{lang}.dev"
        corrected_dev_file = Path(corrected_dir) / f"{lang}.dev"
        devtest_file = Path(devtest_dir) / f"{lang}.devtest"

        print(f"\n{'='*50}")
        print(f"Processing language: {lang}")
        print(f"{'='*50}")

        if not devtest_file.exists():
            print(f"Missing devtest file for language {lang}, skipping...")
            continue

        # Read the translation hypotheses (same for both comparisons)
        hyp = read_file(devtest_file)
        
        # Read source sentences (assuming English source exists)
        src_file = Path(original_dir) / "eng_Latn.dev"  # Adjust path as needed
        if src_file.exists():
            src = read_file(src_file)
        else:
            # If no separate source file, use the original dev as source
            src = read_file(original_dev_file) if original_dev_file.exists() else hyp

        original_scores = None
        corrected_scores = None

        # Evaluate with original dataset
        if original_dev_file.exists():
            ref_original = read_file(original_dev_file)
            
            # Ensure all arrays have same length
            min_len = min(len(ref_original), len(hyp), len(src))
            if len(ref_original) != len(hyp) or len(src) != len(hyp):
                print(f"[WARN] Length mismatch in {lang}: truncating to {min_len}")
                ref_original = ref_original[:min_len]
                hyp_original = hyp[:min_len]
                src_original = src[:min_len]
            else:
                hyp_original = hyp
                src_original = src

            original_scores = evaluate_metrics(ref_original, hyp_original, src_original, lang, 
                                             comet_model, nllb_tokenizer, nllb_model, nllb_device, "Original")

        # Evaluate with corrected dataset
        if corrected_dev_file.exists():
            ref_corrected = read_file(corrected_dev_file)
            
            # Ensure all arrays have same length
            min_len = min(len(ref_corrected), len(hyp), len(src))
            if len(ref_corrected) != len(hyp) or len(src) != len(hyp):
                print(f"[WARN] Length mismatch in {lang}: truncating to {min_len}")
                ref_corrected = ref_corrected[:min_len]
                hyp_corrected = hyp[:min_len]
                src_corrected = src[:min_len]
            else:
                hyp_corrected = hyp
                src_corrected = src

            corrected_scores = evaluate_metrics(ref_corrected, hyp_corrected, src_corrected, lang, 
                                              comet_model, nllb_tokenizer, nllb_model, nllb_device, "Corrected")

        # Compare results
        if original_scores and corrected_scores:
            compare_datasets(original_scores, corrected_scores, lang)
        elif not original_dev_file.exists():
            print(f"Original dataset not found for {lang}")
        elif not corrected_dev_file.exists():
            print(f"Corrected dataset not found for {lang}")

if __name__ == "__main__":
    # Update these paths based on your directory structure
    original_dir = "dev"          # Original FLORES dev files
    corrected_dir = "Corr_devnull"        # Corrected FLORES dev files (from GitHub repo)
    devtest_dir = "devtest"                       # Your translation outputs to evaluate
    
    # Download the corrected dataset first:
    # git clone https://github.com/dsfsi/flores-fix-4-africa.git
    # Then update the corrected_dir path accordingly
    
    evaluate_selected_languages_comparison(original_dir, corrected_dir, devtest_dir)