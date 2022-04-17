from django.db import models

class Predictions(models.Model):

    prediction_time = models.TimeField()
    image_link = models.URLField(max_length = 200)
    image_label = models.CharField(max_length = 50)
    