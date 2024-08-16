import random
import metric_helper as mH
import metrics as M

def get_assert(output, context):

    model = output["model"]

    m = M.Metrics()

    #print("OUTPUTJUNGE!!!:", output["keywords"])

    score = m.calc_keyword_score(output["output"], output["keywords"])    

    return mH.metricHelper.get_metric_output(output, context, model, "KeywordScore", score,{}, threshold=0.0)