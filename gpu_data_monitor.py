import pynvml

# A simple nvidia gpu monitor with pynvml library
# It will print errors
# Data monitored is limited by restrictions of devices
class NVMLClient:
    def __init__(self):
        self.nvmlflag = True
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            print(f"NVIDIA GPU number: {count}")
        except pynvml.NVMLError as e:
            print(f"Cannot get access to NVIDIA GPU: {e}")
            self.nvmlflag = False

    def list_gpus(self):
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = {
                'gpu_index' : i,
                'name': pynvml.nvmlDeviceGetName(handle),
                'util_gpu': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                'util_mem': pynvml.nvmlDeviceGetUtilizationRates(handle).memory,
                'mem_total': pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2),
                'mem_used': pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2),
                'mem_free': pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024**2),
                'temperature': pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            }
            gpus.append(info)
        return gpus

    def shutdown(self):
        pynvml.nvmlShutdown()