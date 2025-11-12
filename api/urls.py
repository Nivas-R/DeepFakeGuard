from django.urls import path
from . import views

urlpatterns = [
    path('ping/', views.ping, name='ping'),
    path('analyze_image/', views.analyze_image, name='analyze_image'),
    path('analyze_video/', views.analyze_video, name='analyze_video'),
    path('analyze_audio/', views.analyze_audio, name='analyze_audio'),
    path('analyze_text/', views.analyze_text, name='analyze_text'),
]
