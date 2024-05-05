class metricHelper:

    @staticmethod
    def get_metric_output(output, context, model, name, score, pricing, threshold):

        metric = {
            "model": model,
            "name": name,
            "data": {
                "score": score,
                "pricing": pricing
            },
            "testContext": context
        }

        return {
            'pass': bool(score >= threshold),
            'score': score,
            'reason': "",
            'metric': metric
        }