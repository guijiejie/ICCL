class Registery(object):
    def __init__(self, module_name):
        super().__init__()
        self._dict = {}
        self.module_name = module_name
    
    def register(self, target):
        def add_register_item(key, value):
            if not callable(value):
                raise Exception(f"register object must be callable! But receice:{value} is not callable!")
            if key in self._dict:
                print(f"warning: {value.__name__} has been registered before, so we will override it")
            self._dict[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            return lambda x : add_register_item(target, x)
    
    def load(self, name):
        if name not in self._dict:
            raise Exception(f"please check whether class [{name}] has been registered in module [{self.module_name}]! The keys in {self.module_name}:{self._dict}")
        
        return self._dict[name]
    
    def contain(self, name):
        return name in self._dict