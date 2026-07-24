from abc import abstractmethod


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, device=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_cuda(self):
        """
        Send all models to self.device.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_cpu(self):
        """
        Send all models to cpu.
        """
        raise NotImplementedError()
