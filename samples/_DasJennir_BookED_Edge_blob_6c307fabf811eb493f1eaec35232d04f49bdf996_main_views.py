import users
import datetime as dt  # extra
from newsapi import NewsApiClient  # extra
from .models import Post, School, Semester, Course, Report_User
from .models import Class as Classes  # change name because of compatibility
from .filters import PostFilter
from .forms import PostCreateForm
from django.shortcuts import (
    render,
    redirect,
    get_object_or_404,
)  # 404 page display, redirect if a action happends, render page
from django.contrib.auth.models import User
from django.contrib.auth.mixins import (
    LoginRequiredMixin,
    UserPassesTestMixin,
)  # auth required
from django.contrib.messages.views import SuccessMessageMixin  # alerts pop up
from django.contrib import messages  # alerts pop up
from django.core.paginator import Paginator  # email
from django.core.mail import send_mail  # email
from django.conf import settings  # email
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.urls import reverse_lazy
from django.views.generic import (
    ListView,
    DetailView,
    CreateView,
    UpdateView,
    DeleteView,
)  # logic of post


# News API
newsapi = NewsApiClient("c368efff5ae140c896773ec0e2dcae10")
data = newsapi.get_everything(
    q="education canada",
    language="en",
    page_size=20,
    sort_by="relevancy",
    domains="theglobeandmail.com",
)

# logic for news page
def home(request):
    context = {"articles": data["articles"]}
    return render(request, "main/home.html", context)


def is_calid_queryparam(param):
    return param != "" and param is not None


# start of filter/find your book
def filters(request):

    if not request.user.is_authenticated:  # return to main page if it is anonymous

        messages.warning(
            request, f"Please create a account to use the filter feature !!!"
        )  # alert menssage
        return redirect("login")

    elif request.user.profile.semester == None:

        messages.warning(
            request,
            f"Please specify the semester in which you are currently studying !!!",
        )
        return redirect("profile")

    else:
        schools_all = request.user.profile.school
        course_all = request.user.profile.course
        classes_all = request.user.profile.classes
        semester_all = request.user.profile.semester
        # schools_all = School.objects.order_by('name') #alphabetical order
        # course_all = Course.objects.order_by('name') #alphabetical order
        # classes_all = Classes.objects.order_by('name') #alphabetical order
        qs = Post.objects.filter(
            visible=True
        )  # Post.objects.all().order_by('-date_posted')
        title_contains_query = request.GET.get("title_contains")
        isbn_query = request.GET.get("title_or_author")
        semester_query = request.GET.get("semester")
        date_min = request.GET.get("date_min")
        date_max = request.GET.get("date_max")
        schools_query = request.GET.get("schools")
        sponsored_query = request.GET.get("sponsored")
        classes_query = request.GET.get("classes")
        course_query = request.GET.get("course")

        # only display books that matches the queries

        if is_calid_queryparam(title_contains_query):
            qs = qs.filter(title__icontains=title_contains_query)

        if is_calid_queryparam(isbn_query):
            qs = qs.filter(isbn__icontains=isbn_query)

        if is_calid_queryparam(semester_query):
            qs = qs.filter(semester=semester_query)

        if is_calid_queryparam(date_min):
            qs = qs.filter(date_posted__gte=date_min)

        if is_calid_queryparam(date_max):
            qs = qs.filter(date_posted__lt=date_max)

        if (
            is_calid_queryparam(schools_query)
            and schools_query != "Institution You are Engaged"
        ):
            qs = qs.filter(schools__name=schools_query)

        if (
            is_calid_queryparam(classes_query)
            and classes_query != "Class You are Engaged"
        ):
            qs = qs.filter(classes__name=classes_query)

        if (
            is_calid_queryparam(course_query)
            and course_query != "Course You are Engaged"
        ):
            qs = qs.filter(course__name=course_query)

        if sponsored_query == "on":
            qs = qs.filter(sponsored=True)

        filtered_post = PostFilter(request.GET, queryset=qs)

        paginated_filtered_posts = Paginator(filtered_post.qs, 4)
        page_number = request.GET.get("page")
        post_page_obj = paginated_filtered_posts.get_page(page_number)
        # gotta put here everthing that goes to the html page
        context = {
            "queryset": qs,
            "schools_all": schools_all,
            "course_all": course_all,
            "classes_all": classes_all,
            "semester_all": semester_all,
            "post_page_obj": post_page_obj,
        }
        return render(request, "main/filters.html", context)


# logic for marketplace view and ordering of posts
class PostListView(ListView):
    model = Post
    queryset = Post.objects.filter(visible=True)
    template_name = "main/market.html"  # <app>/<models>_<viewtype>.html <APP>
    context_object_name = "posts"  # <MODELS>
    ordering = ["-date_posted"]
    paginate_by = 3


# makes pagination of user_posts
class UserPostListView(ListView):
    model = Post
    template_name = "main/user_posts.html"  # <app>/<models>_<viewtype>.html <APP>
    context_object_name = "posts"  # <MODELS>
    ordering = ["-date_posted"]
    paginate_by = 3

    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get("username"))
        return Post.objects.filter(author=user).order_by("-date_posted")


class PostDetailView(DetailView):  # show /post/<number> pages
    model = Post


class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    success_url = "/"

    def test_func(self):  # blocks user from editing posts that are not theirs
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False


class PostCreateView(
    LoginRequiredMixin, CreateView
):  # sets up form to create new post /post/new
    form_class = PostCreateForm
    template_name = "main/post_form.html"

    def form_valid(self, form):
        user_email = (
            self.request.user.email
        )  # get user email, and send email confirming post if valid
        send_mail(
            "You post has been created succesfuly !!!",
            "You post has been created succesfuly !!!",
            "booked.reset@gmail.com",
            [user_email],
            fail_silently=False,
        )

        form.instance.author = self.request.user  # get users name to put on the post
        return super().form_valid(form)


class PostUpdateView(
    LoginRequiredMixin, UserPassesTestMixin, UpdateView, HttpResponseRedirect
):
    model = Post
    fields = [
        "title",
        "content",
        "schools",
        "semester",
        "isbn",
        "post_img",
        "course",
        "classes",
        "visible",
        "author",
    ]
    # def form_valid(self, form):
    # form.instance.author =  form.author   #self.request.user #get users name to put on the post
    # return super().form_valid(form)
    # form.instance.author =  form.author   #self.request.user #get users name to put on the post

    def test_func(self):  # blocks user from editing posts that are not theirs
        post = self.get_object()
        if self.request.user.is_anonymous:
            return redirect("login")
        else:
            if self.request.user == post.author:
                return True
            elif self.request.user != post.author:
                username = self.request.user  # get username
                title = post.title
                url = f"http://127.0.0.1:8000/post/{post.id}/"
                user_email = (
                    post.author.email
                )  # get user email, and send email confirming post if valid
                user_buy_email = self.request.user.email
                send_mail(
                    f"{username} wants to buy your book '{title}'' !!!",
                    f"I want to buy your book !!!, my email contact is {user_buy_email}, url to book {url}",
                    "booked.reset@gmail.com",
                    [user_email],
                    fail_silently=False,
                )

    def handle_no_permission(self):
        return redirect("profile")  # FIX THIS


class ReportCreateView(SuccessMessageMixin, LoginRequiredMixin, CreateView):

    model = Report_User
    template_name = "main/report_user_form.html"
    fields = ["short_explanation", "content"]
    success_message = "Report Has been Forwarded to a Admin, thanks for making BookED a better place !!!"

    def form_valid(self, form):
        next = self.request.POST.get("next")  # This keeps the post URL in memory
        form.instance.url_report = next  # solution https://stackoverflow.com/questions/51509419/how-to-get-the-url-of-the-page-in-a-createview-of-the-report-flag-user-objectur?noredirect=1&lq=1
        form.instance.author = self.request.user  # get users name to put on the post
        super().form_valid(form)
        return super().form_valid(form)
