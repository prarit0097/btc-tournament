from django.urls import path

from . import views

urlpatterns = [
    path('btc/', views.dashboard, name='btc_dashboard'),
    path('api/btc/price', views.api_price, name='btc_price'),
    path('api/btc/price_at', views.api_price_at, name='btc_price_at'),
    path('api/btc/tournament/summary', views.api_tournament_summary, name='btc_tournament_summary'),
    path('api/btc/tournament/scoreboard', views.api_scoreboard, name='btc_tournament_scoreboard'),
    path('api/btc/tournament/run', views.api_tournament_run, name='btc_tournament_run'),
    path('api/btc/tournament/run/status', views.api_tournament_run_status, name='btc_tournament_run_status'),
    path('api/btc/prediction/latest', views.api_prediction_latest, name='btc_prediction_latest'),
    path('api/btc/prediction/refresh', views.api_prediction_refresh, name='btc_prediction_refresh'),
]
