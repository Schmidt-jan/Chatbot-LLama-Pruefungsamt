import random
import metric_helper as mH
import metrics as M

def get_assert(output, context):

    model = output["model"]

    m = M.Metrics()

    score = m.calc_rouge2_score(output["output"], output["answer"])    

    return mH.metricHelper.get_metric_output(output, context, model, "Rouge2", score,{}, threshold=0.0)