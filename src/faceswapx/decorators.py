from .conf.settings import settings


def cpu_offload(func):
    def wrapper(self, *args, **kwargs):
        current_device = self.get_device()
        if current_device != settings.device:
            self.set_device(settings.device)
        result = func(self, *args, **kwargs)

        if settings.CPU_OFFLOAD:
            self.set_device('cpu')

        return result

    return wrapper
