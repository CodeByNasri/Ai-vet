#!/usr/bin/env python3
"""
Analyze trained model files and extract metadata
"""

import torch
import os
from pathlib import Path
from datetime import datetime

def analyze_model_file(model_path):
    """Analyze a single model file"""
    if not Path(model_path).exists():
        return None
    
    try:
        # Get file info
        file_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(Path(model_path).stat().st_mtime)
        
        # Load model
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Analyze parameters
        total_params = 0
        layer_info = {}
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                param_count = param.numel()
                total_params += param_count
                layer_info[name] = {
                    'shape': list(param.shape),
                    'params': param_count,
                    'dtype': str(param.dtype)
                }
        
        return {
            'file_size_mb': round(file_size, 2),
            'modified': mod_time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_parameters': total_params,
            'num_layers': len(state_dict),
            'layer_info': layer_info
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    print("🔍 ANALYZING TRAINED MODELS")
    print("=" * 50)
    
    model_files = [
        'best_classification_model.pth',
        'best_hoofed_animals_model.pth', 
        'best_weight_model.pth',
        'classification_model_final.pth',
        'hoofed_animals_model_final.pth'
    ]
    
    results = {}
    
    for model_file in model_files:
        print(f"\n📁 Analyzing {model_file}...")
        analysis = analyze_model_file(model_file)
        
        if analysis is None:
            print(f"❌ File not found")
            results[model_file] = None
        elif 'error' in analysis:
            print(f"❌ Error: {analysis['error']}")
            results[model_file] = analysis
        else:
            print(f"✅ File size: {analysis['file_size_mb']} MB")
            print(f"✅ Modified: {analysis['modified']}")
            print(f"✅ Total parameters: {analysis['total_parameters']:,}")
            print(f"✅ Number of layers: {analysis['num_layers']}")
            
            # Show first few layers
            layer_names = list(analysis['layer_info'].keys())[:3]
            print(f"✅ Sample layers: {layer_names}")
            
            results[model_file] = analysis
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    available_models = [k for k, v in results.items() if v is not None and 'error' not in v]
    print(f"Available models: {len(available_models)}")
    for model in available_models:
        if results[model]:
            print(f"  - {model}: {results[model]['total_parameters']:,} parameters")
    
    return results

if __name__ == "__main__":
    main()
