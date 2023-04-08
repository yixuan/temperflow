opts = dict(jit=False, debug=False)

def set_opts(jit=False, debug=False):
    opts["jit"] = jit
    opts["debug"] = debug
    return opts
