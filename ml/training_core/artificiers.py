from .arch import Artificier, ArtificierArguments



class ArtifactRemover(Artificier):
    def __init__(self, *fields_to_remove: str):
        self.fields_to_remove = fields_to_remove

    def run(self, args: ArtificierArguments):
        for field in self.fields_to_remove:
            setattr(args.result, field, None)

