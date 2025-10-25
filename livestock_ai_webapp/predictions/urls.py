from django.urls import path
from . import views
from . import debug_views
from . import comprehensive_views

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('weight/', views.WeightPredictionView.as_view(), name='weight_prediction'),
    path('classification/', views.ClassificationView.as_view(), name='classification'),
    path('hoofed-animals/', views.HoofedAnimalsView.as_view(), name='hoofed_animals'),
    path('disease/', views.DiseaseDetectionView.as_view(), name='disease_detection'),
    path('history/', views.HistoryView.as_view(), name='history'),
    path('delete/<int:prediction_id>/', views.delete_prediction, name='delete_prediction'),
    # Mobile App URLs
    path('mobile/', comprehensive_views.ComprehensiveAnalysisView.as_view(), name='mobile_app'),
    path('comprehensive-analysis/', comprehensive_views.ComprehensiveAnalysisView.as_view(), name='comprehensive_analysis'),
    # Debug URLs
    path('debug-weight/', debug_views.DebugWeightView.as_view(), name='debug_weight'),
    path('debug-info/', debug_views.debug_info, name='debug_info'),
]
