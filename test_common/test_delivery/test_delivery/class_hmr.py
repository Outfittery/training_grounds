

class _FeaturizerForTestPurposes:
    def __init__(self, return_value):
        self.return_value = return_value
        
    def get_suffix(self):
        return ', package '

    def __call__(self, *args, **kwargs):
        return self.return_value+self.get_suffix()
