from django.db import models

class PredictionResult(models.Model):
    """Store prediction results for analysis"""
    image_name = models.CharField(max_length=255)
    prediction_type = models.CharField(max_length=50)  # weight, classification, disease
    result = models.JSONField()
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
