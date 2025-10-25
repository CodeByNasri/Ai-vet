from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import TemplateView
from .ml_models import ModelManager
import json

@method_decorator(csrf_exempt, name='dispatch')
class ComprehensiveAnalysisView(TemplateView):
    """Comprehensive analysis using all models"""
    template_name = 'mobile_app.html'
    
    def post(self, request, *args, **kwargs):
        print(f"🔍 DEBUG: Comprehensive analysis POST request received")
        print(f"🔍 DEBUG: Files in request: {list(request.FILES.keys())}")
        
        if 'image' not in request.FILES:
            print("🔍 DEBUG: No image file in request")
            return JsonResponse({
                'status': 'error',
                'error': 'No image file provided'
            })
        
        image_file = request.FILES['image']
        print(f"🔍 DEBUG: Image file: {image_file.name}, size: {image_file.size}")
        
        try:
            # Initialize model manager
            print("🔍 DEBUG: Initializing ModelManager...")
            model_manager = ModelManager()
            print(f"🔍 DEBUG: Models loaded: {list(model_manager.models.keys())}")
            
            results = {}
            
            # 1. Weight Estimation
            print("🔍 DEBUG: Running weight estimation...")
            try:
                weight_result = model_manager.predict_weight(image_file)
                if weight_result.get('status') == 'success':
                    results['weight'] = weight_result.get('predicted_weight')
                    results['weight_confidence'] = weight_result.get('confidence')
                else:
                    results['weight'] = 'Error'
                    results['weight_error'] = weight_result.get('error', 'Unknown error')
            except Exception as e:
                print(f"🔍 DEBUG: Weight estimation error: {e}")
                results['weight'] = 'Error'
                results['weight_error'] = str(e)
            
            # 2. Animal Classification
            print("🔍 DEBUG: Running animal classification...")
            try:
                classification_result = model_manager.predict_classification(image_file)
                if classification_result.get('status') == 'success':
                    results['animal_type'] = classification_result.get('predicted_class')
                    results['confidence'] = classification_result.get('confidence')
                    results['class_probabilities'] = classification_result.get('class_probabilities', {})
                else:
                    results['animal_type'] = 'Error'
                    results['classification_error'] = classification_result.get('error', 'Unknown error')
            except Exception as e:
                print(f"🔍 DEBUG: Classification error: {e}")
                results['animal_type'] = 'Error'
                results['classification_error'] = str(e)
            
            # 3. Disease Detection (only if it's cattle)
            results['disease_detected'] = False
            results['disease_type'] = None
            results['disease_confidence'] = None
            
            if results.get('animal_type') == 'Cattle':
                print("🔍 DEBUG: Running disease detection for cattle...")
                try:
                    disease_result = model_manager.predict_disease(image_file)
                    if disease_result.get('status') == 'success':
                        predicted_class = disease_result.get('predicted_class')
                        disease_confidence = disease_result.get('confidence')
                        
                        if predicted_class == 'Diseased':
                            results['disease_detected'] = True
                            results['disease_type'] = 'Lumpy Skin Disease'
                            results['disease_confidence'] = disease_confidence
                        else:
                            results['disease_detected'] = False
                    else:
                        results['disease_error'] = disease_result.get('error', 'Unknown error')
                except Exception as e:
                    print(f"🔍 DEBUG: Disease detection error: {e}")
                    results['disease_error'] = str(e)
            else:
                print(f"🔍 DEBUG: Skipping disease detection - animal type is {results.get('animal_type')}")
            
            # 4. Hoofed Animals Analysis (additional info)
            print("🔍 DEBUG: Running hoofed animals analysis...")
            try:
                hoofed_result = model_manager.predict_hoofed_animals(image_file)
                if hoofed_result.get('status') == 'success':
                    results['hoofed_analysis'] = hoofed_result.get('class_predictions', {})
                else:
                    results['hoofed_error'] = hoofed_result.get('error', 'Unknown error')
            except Exception as e:
                print(f"🔍 DEBUG: Hoofed animals error: {e}")
                results['hoofed_error'] = str(e)
            
            print(f"🔍 DEBUG: Comprehensive analysis completed: {results}")
            
            return JsonResponse({
                'status': 'success',
                'weight': results.get('weight', 'N/A'),
                'animal_type': results.get('animal_type', 'N/A'),
                'confidence': results.get('confidence', 'N/A'),
                'disease_detected': results.get('disease_detected', False),
                'disease_type': results.get('disease_type', 'None'),
                'disease_confidence': results.get('disease_confidence', 'N/A'),
                'analysis_time': 'Just now',
                'models_used': ['Weight Estimation', 'Animal Classification', 'Disease Detection', 'Hoofed Animals']
            })
            
        except Exception as e:
            print(f"🔍 DEBUG: Comprehensive analysis exception: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'status': 'error',
                'error': str(e)
            })
