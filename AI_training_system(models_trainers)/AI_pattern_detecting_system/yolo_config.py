"""
YOLO Configuration
Comprehensive configuration for YOLO-based pattern detection system
"""

YOLO_CONFIG = {
    # Model settings
    'model_path': 'chart_pattern/candlestick_yolo_12x.pt',
    'model_type': 'yolov8',  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    'input_size': (640, 640),
    
    # Detection thresholds
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'max_detections': 100,
    
    # Training configuration
    'training': {
        'epochs': 100,
        'batch_size': 16,
        'imgsz': 640,
        'patience': 20,
        'device': 'cuda',  # cuda, cpu, mps
        'workers': 8,
        'optimizer': 'AdamW',  # SGD, Adam, AdamW
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    },
    
    # Dataset configuration
    'dataset': {
        'path': 'data/patterns',
        'train_split': 0.8,
        'val_split': 0.15,
        'test_split': 0.05,
        'min_samples_per_class': 10,
        'augmentation': True,
        'resize_strategy': 'letterbox'
    },
    
    # Pattern classes
    'classes': {
        'num_classes': 10,
        'names': [
            'head_and_shoulders',
            'inverse_head_and_shoulders',
            'double_top',
            'double_bottom',
            'triangle_ascending',
            'triangle_descending',
            'triangle_symmetrical',
            'flag_bullish',
            'flag_bearish',
            'cup_and_handle'
        ]
    },
    
    # Data augmentation
    'augmentation': {
        'enable': True,
        'rotation': 0.0,
        'horizontal_flip': 0.5,
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': 0.3,
        'hue': 0.03,
        'blur': 0.0,
        'noise': 0.0
    },
    
    # Inference settings
    'inference': {
        'device': 'cuda',
        'half': False,  # Use FP16 for faster inference
        'agnostic_nms': False,
        'retina_masks': False,
        'verbose': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': 2
    },
    
    # Export settings
    'export': {
        'formats': ['onnx', 'torchscript', 'tensorrt'],
        'imgsz': 640,
        'keras': False,
        'optimize': False,
        'int8': False,
        'dynamic': False,
        'simplify': False,
        'opset': 12
    },
    
    # Visualization settings
    'visualization': {
        'line_width': 2,
        'box_color': 'yellow',
        'label_color': 'white',
        'font_size': 12,
        'show_confidence': True,
        'show_class_name': True
    },
    
    # Performance optimization
    'optimization': {
        'enable_amp': True,  # Automatic Mixed Precision
        'compile': True,  # Torch compile for faster inference
        'dnn': True,  # Use OpenCV DNN
        'optimize_model': False  # Model optimization
    },
    
    # Logging and monitoring
    'logging': {
        'level': 'INFO',
        'save_dir': 'runs/detect',
        'save_period': 10,
        'plots': True,
        'val': True
    },
    
    # Checkpoints
    'checkpoint': {
        'save_period': 10,
        'best_only': False,
        'save_optimizer': True,
        'save_dir': 'weights'
    },
    
    # Real-time settings
    'real_time': {
        'stream': False,
        'fps': 30,
        'buffer_size': 30,
        'display': True
    }
}

