from rest_framework import serializers
from rest_framework.serializers import *
from inbox.models import Notification, NotificationItem
from authentication.models import FollowRequest, Author
from posts.models import Post, Comment, LikeComment, LikePost, Share
from posts.serializers import PostSerializer, CommentSerializer, LikePostSerializer, LikeCommentSerializer
from authentication.serializers import AuthorSerializer, FollowRequestSerializer
from utils import build_default_author_uri
from urllib.parse import urlparse
from authentication.models import Node
from requests.auth import HTTPBasicAuth
import requests
from django.contrib.contenttypes.models import ContentType


class NotificationItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = NotificationItem
        fields = ("__all__")

    # Michael Van De Waeter, handling polymorhic types, October 23, 2023
    # https://stackoverflow.com/questions/19976202/django-rest-framework-django-polymorphic-modelserialization
    def to_representation(self, instance):
        if isinstance(instance.content_object, FollowRequest):
            return FollowRequestSerializer(instance=instance.content_object, context=self.context).data
        if isinstance(instance.content_object, Post):
            return PostSerializer(instance=instance.content_object, context=self.context).data
        if isinstance(instance.content_object, Comment):
            return CommentSerializer(instance=instance.content_object, context=self.context).data
        if isinstance(instance.content_object, LikePost):
            return LikePostSerializer(instance=instance.content_object, context=self.context).data
        if isinstance(instance.content_object, LikeComment):
            return LikeCommentSerializer(instance=instance.content_object, context=self.context).data
        if instance.object_url:
            nodes = Node.objects.all()
            if instance.content_type == ContentType.objects.get_for_model(Post):
                for node in nodes:
                    if node.baseUrl in instance.object_url:
                        auth = HTTPBasicAuth(
                            node.ourUsername, node.ourPassword)
                        response = requests.get(instance.object_url, auth=auth, headers={
                                                "origin": self.context["request"].META["HTTP_HOST"]})

                        return response.json()
            elif instance.content_type == ContentType.objects.get_for_model(LikeComment):
                return LikeCommentSerializer(instance=instance.content_object, context=self.context).data
            return None


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
