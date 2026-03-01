from .nodes import SystematicImageLoader, ImageListStepper, AutoQueueNextRun

NODE_CLASS_MAPPINGS = {
    "SystematicImageLoader": SystematicImageLoader,
    "ImageListStepper": ImageListStepper,
    "AutoQueueNextRun": AutoQueueNextRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SystematicImageLoader": "Systematic Image Loader (Folder Iterator)",
    "ImageListStepper": "Image List Stepper (Advance/Jump/Reset)",
    "AutoQueueNextRun": "Auto Queue Next Run (Loop)",
}
