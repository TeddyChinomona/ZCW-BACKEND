from django.contrib import admin
from .models import *

admin.site.register(CustomUser)
admin.site.register(BaseStation)
admin.site.register(Station)
admin.site.register(CrimeType)
admin.site.register(CrimeIncident)
