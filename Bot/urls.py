from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.index, name='index'),
    path('', views.login_view, name='login_view'),
    path('register/', views.register, name='register'),
]
