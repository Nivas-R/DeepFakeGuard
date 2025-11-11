# api/urls.py
from django.urls import path
from . import views  # import your functions from views.py

urlpatterns = [
    path('analyze_image/', views.analyze_image, name='analyze_image'),
    path('ping/', views.ping, name='ping'),  # simple test endpoint
]
