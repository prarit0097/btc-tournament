import requests
from django.http import JsonResponse
from django.shortcuts import render

BINANCE_TICKER_URL = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'


def index(request):
    return render(request, 'index.html')


def api_price(request):
    try:
        resp = requests.get(
            BINANCE_TICKER_URL,
            timeout=4,
            headers={'User-Agent': 'btc-price-demo/1.0'},
        )
        resp.raise_for_status()
        data = resp.json()
        amount = data.get('price')
        return JsonResponse({'ok': True, 'amount': amount})
    except Exception as exc:
        return JsonResponse({'ok': False, 'error': str(exc)}, status=502)
