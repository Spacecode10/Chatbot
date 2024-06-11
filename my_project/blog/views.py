from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def index(request):
  return render(request, "blog/index.html")

from django.http import JsonResponse

def getResponse(request):
    userMessage = request.GET.get('userMessage')
    return HttpResponse(userMessage)
