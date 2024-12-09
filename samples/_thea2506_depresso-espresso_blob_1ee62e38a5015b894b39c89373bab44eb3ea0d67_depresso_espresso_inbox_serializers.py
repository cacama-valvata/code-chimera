from rest_framework import serializers
from rest_framework.serializers import *
from inbox.models import Notification, NotificationItem
from authentication.models import FollowRequest, Author
from posts.models import Post, Comment, LikeComment, LikePost, Share
from posts.serializers import PostSerializer, CommentSerializer
from authentication.serializers import AuthorSerializer, FollowRequestSerializer
from utils import build_default_author_uri


class NotificationItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = NotificationItem
        fields = ("__all__")

    # Michael Van De Waeter, handling polymorhic types, October 23, 2023
    # https://stackoverflow.com/questions/19976202/django-rest-framework-django-polymorphic-modelserialization
    def to_representation(self, instance):
        if isinstance(instance.content_object, FollowRequest):
            return FollowRequestSerializer(instance=instance.content_object, context=self.context).data


class NotificationSerializer(serializers.ModelSerializer):
    author = SerializerMethodField("get_author_url")
    items = SerializerMethodField("get_items")
    type = SerializerMethodField("get_type")

    class Meta:
        model = Notification
        fields = ("type", "author", "items")

    def get_author_url(self, obj):
        return build_default_author_uri(obj=obj, request=self.context["request"], source="inbox")

    def get_items(self, obj):
        return NotificationItemSerializer(obj.items.all().order_by("-id"), many=True, context=self.context).data

    def get_type(self, _):
        return "inbox"
