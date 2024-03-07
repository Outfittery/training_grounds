from yo_fluq_ds import Obj
from ...single_frame_training import ModelConstructor

class CtorAdapter:
    def __init__(self,
                 type,
                 args_names = (),
                 **kwargs
                 ):
        self.type = type
        self.args_names = args_names
        self.kwargs = Obj(**kwargs)

    def __call__(self, *args, **kwargs):
        if len(args)!=len(self.args_names):
            raise ValueError(f'Expected {len(self.args_names)} argument: {self.args_names}, but received {len(args)}\n{args}')
        final_dictionary = {}
        for arg_name, arg in zip(self.args_names, args):
            final_dictionary[arg_name] = arg
        for key, value in self.kwargs.items():
            final_dictionary[key] = value
        for key, value in kwargs.items():
            final_dictionary[key] = value

        if isinstance(self.type, str):
            type = ModelConstructor._load_class(self.type)
        else:
            type = self.type

        return type(**final_dictionary)
