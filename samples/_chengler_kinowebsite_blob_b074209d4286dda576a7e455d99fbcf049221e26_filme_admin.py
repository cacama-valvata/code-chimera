import os
from django.contrib import admin

from mysite.privat_settings import DEFAULT_DOMAIN
# Register your models here.
from .models import Film,  Inhaltsseite, Event, NewsletterAbonnent, NewsletterSent, Flyer

import logging
from logging import FileHandler
logger = logging.getLogger(__name__) 



class FilmAdmin(admin.ModelAdmin):
    search_fields = ['name', 'status']
    fieldsets = [
        (None,               {'fields': ['name']}),
        ('Löschschutz', {'fields': ['film_no_delete']}),
        ('Beschreibung', {'fields': ['description']}),


    ]
    list_display = ('name', 'status', 'film_no_delete')


class EventAdmin(admin.ModelAdmin):
    search_fields = ['termin', 'kategorie']
    list_display = ('termin', 'film', 'event_online', 'kategorie')
    fieldsets = [
        (None,               {'fields': ['termin']}),
        ('Film', {'fields': ['location_address']}),
    ]

class NewsletterAdmin(admin.ModelAdmin):
    search_fields = ['email',]
    list_display = ('email', 'beantragt', 'opt_in')

# https://stackoverflow.com/questions/1245214/django-admin-exclude-field-on-change-form-only
class FlyerAdmin(admin.ModelAdmin):
    # erstelle eigenes löschmodel
    # https://stackoverflow.com/questions/15196313/django-admin-override-delete-method
    def delete_model(modeladmin, request, queryset):
        for obj in queryset:
            logger.debug("*** delete_model: obj %s ", obj.flyer)
            obj.flyer.delete()  
            obj.delete()
    delete_model.short_description = 'Lösche Datei und Datenbankeintrag'

    # deaktiviere Standard Löschmodel
    # https://stackoverflow.com/questions/1565812/the-default-delete-selected-admin-action-in-django              
    def get_actions(self, request):
        actions = super(FlyerAdmin, self).get_actions(request)
        del actions['delete_selected']
        return actions
     
    list_display = ( 'anzeigename', 'prefix', 'bisZum', 'flyer', 'pk')
    actions = [delete_model]


 

    


admin.site.register(Film, FilmAdmin)
admin.site.register(Inhaltsseite)
admin.site.register(Event, EventAdmin)
admin.site.register(NewsletterAbonnent, NewsletterAdmin)
admin.site.register(NewsletterSent)
admin.site.register(Flyer, FlyerAdmin)






