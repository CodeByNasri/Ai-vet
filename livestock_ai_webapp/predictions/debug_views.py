from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import TemplateView
import json

@method_decorator(csrf_exempt, name='dispatch')
class DebugWeightView(TemplateView):
    """Debug view for weight prediction"""
    template_name = 'predictions/debug_weight.html'
    
    def post(self, request, *args, **kwargs):
        print("🔍 DEBUG: Form submitted!")
        print(f"🔍 DEBUG: Request method: {request.method}")
        print(f"🔍 DEBUG: Content type: {request.content_type}")
        print(f"🔍 DEBUG: Files: {list(request.FILES.keys())}")
        print(f"🔍 DEBUG: POST data: {dict(request.POST)}")
        
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            print(f"🔍 DEBUG: Image file: {image_file.name}, size: {image_file.size}")
            
            # Test the model manager
            try:
                from .ml_models import ModelManager
                model_manager = ModelManager()
                
                if 'weight' in model_manager.models:
                    result = model_manager.predict_weight(image_file)
                    print(f"🔍 DEBUG: Prediction result: {result}")
                    
                    return JsonResponse({
                        'status': 'success',
                        'result': result,
                        'debug_info': {
                            'file_name': image_file.name,
                            'file_size': image_file.size,
                            'models_loaded': list(model_manager.models.keys())
                        }
                    })
                else:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Weight model not loaded'
                    })
            except Exception as e:
                print(f"🔍 DEBUG: Error: {e}")
                import traceback
                traceback.print_exc()
                return JsonResponse({
                    'status': 'error',
                    'message': str(e)
                })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'No image file provided'
            })

def debug_info(request):
    """Debug info endpoint"""
    from .ml_models import ModelManager
    
    try:
        model_manager = ModelManager()
        return JsonResponse({
            'models_loaded': list(model_manager.models.keys()),
            'django_debug': True,
            'media_root': str(model_manager.device)
        })
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        })
