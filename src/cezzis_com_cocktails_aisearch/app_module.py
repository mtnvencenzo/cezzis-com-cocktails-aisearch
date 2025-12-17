from injector import Binder, Module, singleton


class AppModule(Module):
    def configure(self, binder: Binder):
        from mediatr import Mediator

        binder.bind(Mediator, Mediator, scope=singleton)
