from django.shortcuts import render, redirect
from .forms import SignUpForm, LoginForm
from django.contrib.auth import authenticate, login


# Create your views here.


def index(request):
    username = request.user.username  # Assuming the user is authenticated and username is available
    show_sentiment = username == 'mit'
    return render(request, 'chat.html', {'show_sentiment': show_sentiment})



def register(request):
    msg = None
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            msg = 'user created'
            return redirect('login_view')
        else:
            msg = 'form is not valid'
    else:
        form = SignUpForm()
    return render(request, 'register.html', {'form': form, 'msg': msg})


def login_view(request):
    form = LoginForm(request.POST or None)
    msg = None
    if request.method == 'POST':
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                if username == 'mit':
                    return redirect('index')
                else :
                    return redirect('index')
                    # Redirect to admin chatbot page
            else:
                msg = 'Invalid credentials'
        else:
            msg = 'Error validating form'
    return render(request, 'login.html', {'form': form, 'msg': msg})


# def admin(request):
#     return render(request, 'admin.html')


# def customer(request):
#     return render(request, 'customer.html')
