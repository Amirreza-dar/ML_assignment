from django.urls import include, path
from . import views

app_name = 'api'

urlpatterns = [
    path('predict/<str:url>/', views.predict, name = 'predict'),
    path('clear-past-predictions/', views.clear, name = 'clear'),
    path('get_past_predictions/', views.allPredictions, name = 'allPredictions'),
]