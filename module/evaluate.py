def cal_sentiment_prf(tp, fp, fn, num_of_aspect, verbal=False):
    p = [tp[i] / (tp[i] + fp[i]) for i in range(num_of_aspect)]
    r = [tp[i] / (tp[i] + fn[i]) for i in range(num_of_aspect)]
    f1 = [2 * p[i] * r[i] / (p[i] + r[i]) for i in range(num_of_aspect)]

    micro_p = sum(tp) / (sum(tp) + sum(fp))
    micro_r = sum(tp) / (sum(tp) + sum(fn))
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)

    macro_p = sum(p) / num_of_aspect
    macro_r = sum(r) / num_of_aspect
    macro_f1 = sum(f1) / num_of_aspect

    if verbal:
        print('p:', p)
        print('r:', r)
        print('f1:', f1)
        print('micro:', (micro_p, micro_r, micro_f1))
        print('macro:', (macro_p, macro_r, macro_f1))

    output = f"'p': {p}\n'r': {r}\n'f1': {f1}\n'micro': ({micro_p}, {micro_r}, {micro_f1})\n'macro': ({macro_p}, {macro_r}, " \
             f"{macro_f1})".format(p, r, f1, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1)
    outputs = output
    return outputs

# return p, r, f1, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)
