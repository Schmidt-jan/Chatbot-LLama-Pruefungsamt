from typing import List
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.metrics.distance import edit_distance

#nltk.download('stopwords')

class Metrics:
    @staticmethod
    def get_tokenizer():
        return AutoTokenizer.from_pretrained('bert-base-uncased')

    @staticmethod
    def get_model():
        return AutoModel.from_pretrained('bert-base-uncased')

    @staticmethod
    def get_st_model():
        return SentenceTransformer('all-mpnet-base-v2')

    @staticmethod
    def get_stop_words():
        return set(stopwords.words('english')) | set(stopwords.words('german'))

    @staticmethod
    def get_embedding(text, model, tokenizer):
        tokenizer = tokenizer or Metrics.get_tokenizer()
        model = model or Metrics.get_model()
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def calc_cosine_similarity(self, candidate, references, model=None, tokenizer=None):
        """
        Berechnet die Kosinus-Ähnlichkeit zwischen zwei Texten basierend auf ihren Wortvektoren.
        Ein Wert nahe 1 bedeutet eine hohe Ähnlichkeit, während ein Wert nahe 0 oder -1 eine geringe Ähnlichkeit bedeutet.
        """
        embedding1 = self.get_embedding(candidate, model, tokenizer)
        embedding2 = self.get_embedding(references, model, tokenizer)
        return 1 - cosine(embedding1.detach().numpy(), embedding2.detach().numpy())

    @staticmethod
    def calc_jaccard_similarity(candidate, references):
        """
        Berechnet die Jaccard-Ähnlichkeit zwischen zwei Texten, 
        ein Maß für die Übereinstimmung der Wortsets.
        """
        words_t1 = set(candidate.split())
        words_t2 = set(references.split())
        intersection = words_t1.intersection(words_t2)
        union = words_t1.union(words_t2)
        return float(len(intersection)) / len(union)

    @staticmethod
    def calc_bleu_score(candidate, references):
        """
        Berechnet den BLEU-Score für einen Kandidaten-Text und eine Liste von Referenz-Texten.
        Der BLEU-Score bewertet die Übereinstimmung von n-Grammen zwischen dem Kandidaten und den Referenztexten.
        """
        # Berechnung des BLEU-Scores
        bleu_score = sentence_bleu(references, candidate)
        # Kappung des BLEU-Scores bei 1, um sicherzustellen, dass der Wert im Bereich von 0 bis 1 liegt
        capped_bleu_score = min(1.0, bleu_score)

        return capped_bleu_score

    @staticmethod
    def calc_rouge1_score(candidate, references):
        """
        Berechnet den ROUGE-1-Score für einen Kandidaten-Text und eine Liste von Referenz-Texten.
        ROUGE-1 vergleicht die Übereinstimmung der Unigramme zwischen dem Kandidaten und den Referenztexten.
        """
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        rouge1_score = rouge_scorer_instance.score(candidate, references)['rouge1'].fmeasure
        return rouge1_score

    @staticmethod
    def calc_rouge2_score(candidate, references):
        """
        Berechnet den ROUGE-2-Score für einen Kandidaten-Text und eine Liste von Referenz-Texten.
        ROUGE-2 vergleicht die Übereinstimmung der Bigramme zwischen dem Kandidaten und den Referenztexten.
        """
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        rouge2_score = rouge_scorer_instance.score(candidate, references)['rouge2'].fmeasure
        return rouge2_score

    @staticmethod
    def calc_rougeL_score(candidate, references):
        """
        Berechnet den ROUGE-L-Score für einen Kandidaten-Text und eine Liste von Referenz-Texten.
        ROUGE-L ist eine längenbezogene Variante von ROUGE-1 und vergleicht die Übereinstimmung der Unigramme zwischen dem Kandidaten und den Referenztexten.
        """
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL_score = rouge_scorer_instance.score(candidate, references)['rougeL'].fmeasure
        return rougeL_score
    
    @staticmethod
    def calc_edit_distance(candidate, references):
        """
        Berechnet die Edit-Distanz zwischen zwei Texten.
        Die Edit-Distanz ist die minimale Anzahl von Bearbeitungsschritten
        (Einfügen, Löschen, Ersetzen), die erforderlich sind, um text1 in text2 zu transformieren.
        """
        # Berechnung der Edit-Distanz
        edit_dist = edit_distance(candidate, references)
        # Normalisierung auf den Bereich von 0.0 bis 1.0
        normalized_edit_dist = edit_dist / max(len(candidate), len(references))

        return 1.0 - normalized_edit_dist
    
    @staticmethod
    def calc_keyword_score(text: str, keywords: List[str]) -> float:
        """
        Berechnet den Prozentsatz, wie viele Schlüsselwörter im Text enthalten sind.
        """
        print(keywords)
        text = text.lower()
        keywords = [keyword.lower() for keyword in keywords]
        keyword_count = sum(keyword in text for keyword in keywords)
        return keyword_count / len(keywords) if keywords else 0.0
