from django.contrib import admin
from .models import PredictionResult

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ['image_name', 'prediction_type', 'confidence', 'created_at']
    list_filter = ['prediction_type', 'created_at']
    search_fields = ['image_name']
    readonly_fields = ['created_at']
