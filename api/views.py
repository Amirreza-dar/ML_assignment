from urllib import response
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .models import Predictions
from . import model
import json


def train(request):
    try:
        model.train()
        resp = {
            'success' : 'true'
        }
    except:
        resp = {
            'success' : 'false'
        }
    
    resp = json.dumps(resp)
    return JsonResponse(resp, safe = False)

def predict(request, url):
    link = request['img-link']
    label, predict_time = model.predict(link)

    prediction = Predictions(prediction_time = predict_time, image_link = link, image_label = label)
    prediction.save()


    predict_label = {
        "prediction" : label,
    }

    predict_label = json.dumps(predict_label)

    return HttpResponse(predict_label)

def allPredictions(request):
    predicts = list(Predictions.objects.values())
    return JsonResponse(predicts, safe = False)

def clear(request):
    Predictions.objects.all().delete()
    return HttpResponse(status = 204)
