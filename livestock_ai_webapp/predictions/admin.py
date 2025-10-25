from django.contrib import admin
from .models import Prediction

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['prediction_type', 'created_at', 'confidence', 'get_result_summary']
    list_filter = ['prediction_type', 'created_at']
    search_fields = ['prediction_type']
    readonly_fields = ['created_at']
    
    def get_result_summary(self, obj):
        """Display a summary of the prediction result"""
        if obj.prediction_type == 'weight':
            return f"Weight: {obj.result.get('predicted_weight', 'N/A')} kg"
        elif obj.prediction_type == 'classification':
            return f"Class: {obj.result.get('predicted_class', 'N/A')}"
        elif obj.prediction_type == 'disease':
            return f"Status: {obj.result.get('predicted_class', 'N/A')}"
        elif obj.prediction_type == 'hoofed_animals':
            return "Multi-label analysis"
        return "N/A"
    
    get_result_summary.short_description = 'Result Summary'
