# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AlbumInfo(models.Model):
    album_id = models.AutoField(primary_key=True)
    album_name = models.TextField(blank=True, null=True)
    artist_name = models.TextField(blank=True, null=True)
    album_mood = models.TextField(blank=True, null=True)
    album_cover = models.TextField(blank=True, null=True)

    class Meta:
        app_label='main_page'
        managed = False
        db_table = 'album_info'