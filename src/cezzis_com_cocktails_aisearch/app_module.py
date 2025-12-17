
from injector import singleton, Binder, Module

class AppModule(Module):
    def configure(self, binder: Binder):
        from mediatr import Mediator

        binder.bind(Mediator, Mediator, scope=singleton)