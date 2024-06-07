from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login_view'),
    path('register/', views.register, name='register'),
    path('home/', views.register, name='home'),
    path('admin/', views.admin, name='adminpage'),
    path('customer/', views.customer, name='customer'),
]
