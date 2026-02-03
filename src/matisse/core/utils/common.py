def remove_double_parameter(p):
    """Remove double parameters from a list of parameters."""
    listP = p.split(" ")
    paramName = []
    paramsNew = ""
    for elt in listP:
        idx = elt.find("=")
        if elt[0:idx] not in paramName and elt != "":
            paramName.append(elt[0:idx])
            paramsNew = paramsNew + " " + elt
    return paramsNew
