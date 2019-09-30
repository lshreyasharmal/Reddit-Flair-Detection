from __future__ import unicode_literals
import sys
from django.shortcuts import render
from django.http import HttpResponse
import json
from .detector import *


def index(request):
    if request.method == 'GET':
        return render(request, 'flair/index.html', {'url': '', 'result': ''})
    elif request.method == 'POST':
        url_name = request.POST.get('url_name', None)
        result = predict_flair_from_url(url_name)
        return render(request, 'flair/index.html', {'url': url_name, 'result': result})


def analyze(request):
	
	print("heyyyyyyyyyyyyyy")
	comments_vs_flairs = get_comments_vs_flairs()
	upvotes_vs_flairs = get_upvotes_vs_flairs()
	ups = list(upvotes_vs_flairs.values())
	coms = list(comments_vs_flairs.values())
	keys = list(upvotes_vs_flairs.keys())
	flairs = []
	for k in keys:
		flairs.append("\'"+str(k)+"\'")
	print(flairs)
		
	return render(request, 'flair/analyse.html',{'labels': flairs,'ups_data': ups,"comments_data":coms})