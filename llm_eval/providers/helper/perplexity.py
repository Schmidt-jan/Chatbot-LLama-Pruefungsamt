import math

def compute_perplexity(log_probs):
    total_log_prob = 0
    for log_prob in log_probs:
        total_log_prob += log_prob
    perplexity = math.exp(-total_log_prob / len(log_probs))
    return perplexity