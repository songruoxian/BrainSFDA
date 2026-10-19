from .dynamic_model_su_abide import PreModel
# from .heterogeneous_co_training_abide import PreModel

'''subnetwork'''
def build_model(args):
    nodes = args.nodes
    classes = args.classes

    model = PreModel(
        nodes = nodes,
        classes=classes
    )

    return model
