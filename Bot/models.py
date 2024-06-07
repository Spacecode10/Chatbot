from django.db import models
from django.contrib.auth.models import AbstractUser
class Users(AbstractUser):
    is_admin= models.BooleanField('Is admin',default=False)
    is_customer= models.BooleanField('Is customer',default=True)

# Create your models here.
