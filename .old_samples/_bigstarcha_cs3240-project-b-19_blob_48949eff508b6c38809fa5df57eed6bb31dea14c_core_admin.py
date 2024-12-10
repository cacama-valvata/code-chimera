# Resource used: https://stackoverflow.com/questions/57416061/django-heroku-modulenotfounderror-no-module-named-django-heroku
# https://getbootstrap.com/docs/3.4/getting-started/
# https://stackoverflow.com/questions/59264892/modulenotfounderror-no-module-named-bootstrap4

from django.contrib import admin

from .models import Listing, Review


class ListingAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "latitude",
        "longitude",
        "address1",
        "address2",
        "city",
        "state",
        "zipcode",
        "num_rooms",
        "num_bathrooms",
        "rent",
    )
    list_filter = ["zipcode", "rent"]  # can you have multiple filtration criteria?
    search_fields = [
        "name",
        "city",
        "state",
        "zipcode",
        "num_rooms",
        "num_bathrooms",
        "rent",
    ]


class ReviewAdmin(admin.ModelAdmin):
    list_display = ("rating", "review", "date", "listing", "user")
    list_filter = ["rating"]
    search_fields = ["rating", "date", "listing", "user"]


admin.site.register(Listing, ListingAdmin)
admin.site.register(Review, ReviewAdmin)
