# Generated by Django 3.0.2 on 2022-04-17 20:21

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='predictions',
            old_name='label',
            new_name='image_label',
        ),
        migrations.RenameField(
            model_name='predictions',
            old_name='image',
            new_name='image_link',
        ),
        migrations.RenameField(
            model_name='predictions',
            old_name='time',
            new_name='prediction_time',
        ),
    ]
