from django.contrib import admin
from django.urls import path, include
from price.views import index, api_price

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('api/price', api_price, name='api_price'),
    path('', include('btc_dashboard.urls')),
]
