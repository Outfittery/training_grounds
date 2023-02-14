from .arch import Artificier, ArtificierArguments


class ArtifactRemover(Artificier):
    def __init__(self, *fields_to_remove: str):
        self.fields_to_remove = fields_to_remove

    def run_before_storage(self, args: ArtificierArguments):
        for field in self.fields_to_remove:
            setattr(args.result, field, None)


class ResultDFCleaner(Artificier):
    def __init__(self, *prefixes_to_remove):
        self.prefixes_to_remove = prefixes_to_remove

    def run_before_storage(self, args: ArtificierArguments):
        for prefix in self.prefixes_to_remove:
            columns = [c for c in args.result.result_df.columns if c.startswith(prefix)]
            args.result.result_df.drop(columns, axis=1, inplace=True)