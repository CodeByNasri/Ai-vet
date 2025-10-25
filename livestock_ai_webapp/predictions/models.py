from django.db import models
from django.utils import timezone

class Prediction(models.Model):
    """Model to store prediction results"""
    
    PREDICTION_TYPES = [
        ('weight', 'Weight Estimation'),
        ('classification', 'Animal Classification'),
        ('hoofed_animals', 'Hoofed Animals Classification'),
        ('disease', 'Disease Detection'),
    ]
    
    prediction_type = models.CharField(max_length=20, choices=PREDICTION_TYPES)
    image = models.ImageField(upload_to='predictions/')
    result = models.JSONField()
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.get_prediction_type_display()} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
