from pipeline.action_pipeline import ActionRecognitionPipeline

__all__ = ["ActionRecognitionPipeline"]

try:
    from pipeline.memory_pipeline import FrameMemoryPipeline
except ModuleNotFoundError:
    FrameMemoryPipeline = None
else:
    __all__.append("FrameMemoryPipeline")
