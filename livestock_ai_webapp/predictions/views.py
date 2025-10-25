from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import TemplateView
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import json
import os
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tensorflow as tf
from tensorflow import keras
from .models import Prediction
from .ml_models import ModelManager

class HomeView(TemplateView):
    """Home page with model selection"""
    template_name = 'predictions/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['recent_predictions'] = Prediction.objects.all()[:5]
        return context

class WeightPredictionView(TemplateView):
    """Weight estimation prediction"""
    template_name = 'predictions/weight_prediction.html'
    
    def post(self, request, *args, **kwargs):
        print(f"🔍 DEBUG: Weight prediction POST request received")
        print(f"🔍 DEBUG: Files in request: {list(request.FILES.keys())}")
        print(f"🔍 DEBUG: POST data: {dict(request.POST)}")
        
        if 'image' not in request.FILES:
            print("🔍 DEBUG: No image file in request")
            messages.error(request, 'Please select an image file.')
            return render(request, self.template_name, {'error': 'No image file provided', 'debug_info': 'No image file in request'})
        
        image_file = request.FILES['image']
        print(f"🔍 DEBUG: Image file: {image_file.name}, size: {image_file.size}")
        
        try:
            # Initialize model manager
            print("🔍 DEBUG: Initializing ModelManager...")
            model_manager = ModelManager()
            print(f"🔍 DEBUG: Models loaded: {list(model_manager.models.keys())}")
            
            # Make prediction
            print("🔍 DEBUG: Making weight prediction...")
            result = model_manager.predict_weight(image_file)
            print(f"🔍 DEBUG: Prediction result: {result}")
            
            # Check if prediction was successful
            if result.get('status') == 'error':
                print(f"🔍 DEBUG: Prediction failed: {result.get('error')}")
                messages.error(request, f"Prediction error: {result.get('error', 'Unknown error')}")
                return render(request, self.template_name, {'error': result.get('error')})
            
            # Save prediction to database
            print("🔍 DEBUG: Saving prediction to database...")
            
            # Handle confidence field properly
            confidence_value = result.get('confidence', None)
            if confidence_value == 'N/A' or confidence_value is None:
                confidence_value = None
            else:
                try:
                    confidence_value = float(confidence_value)
                except (ValueError, TypeError):
                    confidence_value = None
            
            prediction = Prediction.objects.create(
                prediction_type='weight',
                image=image_file,
                result=result,
                confidence=confidence_value
            )
            print(f"🔍 DEBUG: Prediction saved with ID: {prediction.id}")
            
            messages.success(request, f"Weight prediction completed successfully!")
            return render(request, self.template_name, {
                'result': result,
                'prediction_id': prediction.id,
                'debug_info': f'Success! Weight: {result.get("predicted_weight")} kg, Model: {result.get("model_type")}'
            })
            
        except Exception as e:
            print(f"🔍 DEBUG: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            messages.error(request, f'Error processing image: {str(e)}')
            return render(request, self.template_name, {
                'error': str(e),
                'debug_info': f'Exception: {str(e)}'
            })

class ClassificationView(TemplateView):
    """Animal classification prediction"""
    template_name = 'predictions/classification.html'
    
    def post(self, request, *args, **kwargs):
        print(f"🔍 DEBUG: Classification POST request received")
        print(f"🔍 DEBUG: Files in request: {list(request.FILES.keys())}")
        print(f"🔍 DEBUG: POST data: {dict(request.POST)}")
        
        if 'image' not in request.FILES:
            print("🔍 DEBUG: No image file in request")
            messages.error(request, 'Please select an image file.')
            return render(request, self.template_name, {'error': 'No image file provided', 'debug_info': 'No image file in request'})
        
        image_file = request.FILES['image']
        print(f"🔍 DEBUG: Image file: {image_file.name}, size: {image_file.size}")
        
        try:
            # Initialize model manager
            print("🔍 DEBUG: Initializing ModelManager...")
            model_manager = ModelManager()
            print(f"🔍 DEBUG: Models loaded: {list(model_manager.models.keys())}")
            
            # Make prediction
            print("🔍 DEBUG: Making classification prediction...")
            result = model_manager.predict_classification(image_file)
            print(f"🔍 DEBUG: Prediction result: {result}")
            
            # Check if prediction was successful
            if result.get('status') == 'error':
                print(f"🔍 DEBUG: Prediction failed: {result.get('error')}")
                messages.error(request, f"Prediction error: {result.get('error', 'Unknown error')}")
                return render(request, self.template_name, {'error': result.get('error')})
            
            # Save prediction to database
            print("🔍 DEBUG: Saving prediction to database...")
            
            # Handle confidence field properly
            confidence_value = result.get('confidence', None)
            if confidence_value == 'N/A' or confidence_value is None:
                confidence_value = None
            else:
                try:
                    confidence_value = float(confidence_value)
                except (ValueError, TypeError):
                    confidence_value = None
            
            prediction = Prediction.objects.create(
                prediction_type='classification',
                image=image_file,
                result=result,
                confidence=confidence_value
            )
            print(f"🔍 DEBUG: Prediction saved with ID: {prediction.id}")
            
            messages.success(request, f"Classification completed successfully!")
            return render(request, self.template_name, {
                'result': result,
                'prediction_id': prediction.id,
                'debug_info': f'Success! Class: {result.get("predicted_class")}, Confidence: {result.get("confidence")}%, Model: {result.get("model_type")}'
            })
            
        except Exception as e:
            print(f"🔍 DEBUG: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            messages.error(request, f'Error processing image: {str(e)}')
            return render(request, self.template_name, {
                'error': str(e),
                'debug_info': f'Exception: {str(e)}'
            })

class HoofedAnimalsView(TemplateView):
    """Hoofed animals classification"""
    template_name = 'predictions/hoofed_animals.html'
    
    def post(self, request, *args, **kwargs):
        print(f"🔍 DEBUG: Hoofed animals POST request received")
        print(f"🔍 DEBUG: Files in request: {list(request.FILES.keys())}")
        print(f"🔍 DEBUG: POST data: {dict(request.POST)}")
        
        if 'image' not in request.FILES:
            print("🔍 DEBUG: No image file in request")
            messages.error(request, 'Please select an image file.')
            return render(request, self.template_name, {'error': 'No image file provided', 'debug_info': 'No image file in request'})
        
        image_file = request.FILES['image']
        print(f"🔍 DEBUG: Image file: {image_file.name}, size: {image_file.size}")
        
        try:
            # Initialize model manager
            print("🔍 DEBUG: Initializing ModelManager...")
            model_manager = ModelManager()
            print(f"🔍 DEBUG: Models loaded: {list(model_manager.models.keys())}")
            
            # Make prediction
            print("🔍 DEBUG: Making hoofed animals prediction...")
            result = model_manager.predict_hoofed_animals(image_file)
            print(f"🔍 DEBUG: Prediction result: {result}")
            
            # Check if prediction was successful
            if result.get('status') == 'error':
                print(f"🔍 DEBUG: Prediction failed: {result.get('error')}")
                messages.error(request, f"Prediction error: {result.get('error', 'Unknown error')}")
                return render(request, self.template_name, {'error': result.get('error')})
            
            # Save prediction to database
            print("🔍 DEBUG: Saving prediction to database...")
            
            # Handle confidence field properly
            confidence_value = result.get('confidence', None)
            if confidence_value == 'N/A' or confidence_value is None:
                confidence_value = None
            else:
                try:
                    confidence_value = float(confidence_value)
                except (ValueError, TypeError):
                    confidence_value = None
            
            prediction = Prediction.objects.create(
                prediction_type='hoofed_animals',
                image=image_file,
                result=result,
                confidence=confidence_value
            )
            print(f"🔍 DEBUG: Prediction saved with ID: {prediction.id}")
            
            messages.success(request, f"Hoofed animals classification completed successfully!")
            return render(request, self.template_name, {
                'result': result,
                'prediction_id': prediction.id,
                'debug_info': f'Success! Hoofed animals analysis completed, Model: {result.get("model_type")}'
            })
            
        except Exception as e:
            print(f"🔍 DEBUG: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            messages.error(request, f'Error processing image: {str(e)}')
            return render(request, self.template_name, {
                'error': str(e),
                'debug_info': f'Exception: {str(e)}'
            })

class DiseaseDetectionView(TemplateView):
    """Disease detection prediction"""
    template_name = 'predictions/disease_detection.html'
    
    def post(self, request, *args, **kwargs):
        print(f"🔍 DEBUG: Disease detection POST request received")
        print(f"🔍 DEBUG: Files in request: {list(request.FILES.keys())}")
        print(f"🔍 DEBUG: POST data: {dict(request.POST)}")
        
        if 'image' not in request.FILES:
            print("🔍 DEBUG: No image file in request")
            messages.error(request, 'Please select an image file.')
            return render(request, self.template_name, {'error': 'No image file provided', 'debug_info': 'No image file in request'})
        
        image_file = request.FILES['image']
        print(f"🔍 DEBUG: Image file: {image_file.name}, size: {image_file.size}")
        
        try:
            # Initialize model manager
            print("🔍 DEBUG: Initializing ModelManager...")
            model_manager = ModelManager()
            print(f"🔍 DEBUG: Models loaded: {list(model_manager.models.keys())}")
            
            # Make prediction
            print("🔍 DEBUG: Making disease detection prediction...")
            result = model_manager.predict_disease(image_file)
            print(f"🔍 DEBUG: Prediction result: {result}")
            
            # Check if prediction was successful
            if result.get('status') == 'error':
                print(f"🔍 DEBUG: Prediction failed: {result.get('error')}")
                messages.error(request, f"Prediction error: {result.get('error', 'Unknown error')}")
                return render(request, self.template_name, {'error': result.get('error')})
            
            # Save prediction to database
            print("🔍 DEBUG: Saving prediction to database...")
            
            # Handle confidence field properly
            confidence_value = result.get('confidence', None)
            if confidence_value == 'N/A' or confidence_value is None:
                confidence_value = None
            else:
                try:
                    confidence_value = float(confidence_value)
                except (ValueError, TypeError):
                    confidence_value = None
            
            prediction = Prediction.objects.create(
                prediction_type='disease',
                image=image_file,
                result=result,
                confidence=confidence_value
            )
            print(f"🔍 DEBUG: Prediction saved with ID: {prediction.id}")
            
            messages.success(request, f"Disease detection completed successfully!")
            return render(request, self.template_name, {
                'result': result,
                'prediction_id': prediction.id,
                'debug_info': f'Success! Disease detection completed, Model: {result.get("model_type")}'
            })
            
        except Exception as e:
            print(f"🔍 DEBUG: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            messages.error(request, f'Error processing image: {str(e)}')
            return render(request, self.template_name, {
                'error': str(e),
                'debug_info': f'Exception: {str(e)}'
            })

class HistoryView(TemplateView):
    """View prediction history"""
    template_name = 'predictions/history.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['predictions'] = Prediction.objects.all()
        return context

def delete_prediction(request, prediction_id):
    """Delete a prediction"""
    try:
        prediction = Prediction.objects.get(id=prediction_id)
        prediction.delete()
        messages.success(request, 'Prediction deleted successfully.')
    except Prediction.DoesNotExist:
        messages.error(request, 'Prediction not found.')
    
    return redirect('history')
