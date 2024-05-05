import random
import metric_helper as mH
import metrics as M

def get_assert(output, context):

    model = output["model"]

    m = M.Metrics()

    score = m.calc_bleu_score(output["output"], output["answer"])    

    return mH.metricHelper.get_metric_output(output, context, model, "BleuScore", score,{}, threshold=0.0)