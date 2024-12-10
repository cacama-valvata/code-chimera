from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.auth.models import User
import pandas as pd
import json
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token
from django.conf import settings


class Dictionary(models.Model):
    title = models.CharField(max_length=60)
    source = models.URLField( max_length=400,blank=True,null=True)
    description = models.CharField( max_length=1000,blank=True,null=True)
    file = models.FileField(blank=False, null=False,validators=[FileExtensionValidator(allowed_extensions=["csv"])])
    user = models.ForeignKey(User, related_name = 'dictionaries', on_delete=models.CASCADE)
    public = models.BooleanField(default=False)
    
    # https://stackoverflow.com/questions/2208219/return-function-as-a-field-on-django-model
    @property
    def preview(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None)
        data = data.iloc[:, 0]
        data = data.head(6)
        data = data.values.tolist()
        data = json.dumps(data)
        return (data)
    
    @property
    def words_full(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None, usecols=[0])
        data = data.dropna()
        data = data.values.tolist()
        print(data)
        data = json.dumps(data)
        return (data)
    
    @property
    def words_q1(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None, usecols=[0])
        data = data.dropna()
        data = data.loc[0: int(round(data.shape[0]/4))]
        data = data.values.tolist()
        data = json.dumps(data)
        return (data)
    
    @property
    def words_q2(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None, usecols=[0])
        data = data.dropna()
        data = data.loc[int(round(data.shape[0]/4)+1): int(round(data.shape[0]/4)*2) ]
        data = data.values.tolist()
        data = json.dumps(data)
        return (data)
    

    
    @property
    def words_q3(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None, usecols=[0])
        data = data.dropna()
        data = data.loc[int(round(data.shape[0]/4)*2+1): int(round(data.shape[0]/4)*3) ]
        data = data.values.tolist()
        data = json.dumps(data)
        return (data)
      
    @property
    def words_q4(self):
    # read & prep
        data = pd.read_csv(self.file.url, header=None, usecols=[0])
        data = data.dropna()
        data = data.loc[int(round(data.shape[0]/4)*3+1): int(round(data.shape[0])) ]
        data = data.values.tolist()
        data = json.dumps(data)
        return (data)

    @property
    def filter_df_q_no_u(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None)
        data = data.iloc[:, 0]
        # filter
        data = data[data.str.contains('q', case=False, na=False)]
        data = data[~data.str.contains('u', case=False, na=False)]
        data = data.values.tolist()
        # return
        return json.dumps(data)
    
    @property
    def filter_df_palindrome(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None)
        data = data.iloc[:, 0]
        # filter
        data = pd.DataFrame(data)
        data.columns = ['word']
        data['rev'] = data['word'].str[::-1]
        data = data[data['rev']==data['word']]
        data = data.iloc[:, 0]
        # return
        return json.dumps(data)
    
    @property
    def filter_df_7l5v(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None)
        data = data.iloc[:, 0]
        # filter
        data = data[data.str.len() == 7]
        data = data[data.str.count(r'[aeiouAEIOU]') == 5]
        return json.dumps(data)
    
    @property
    def filter_df_5l4v(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None)
        data = data.iloc[:, 0]
        # filter
        data = data[data.str.len() == 5]
        data = data[data.str.count(r'[aeiouAEIOU]') == 4]
        return json.dumps(data)
    
    @property
    def filter_df_satine(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None)
        data = data.iloc[:, 0]
        # filter
        data = data[data.str.len() == 7]
        data = data[data.str.contains('s', case=False, na=False)]
        data = data[data.str.contains('a', case=False, na=False)]
        data = data[data.str.contains('t', case=False, na=False)]
        data = data[data.str.contains('i', case=False, na=False)]
        data = data[data.str.contains('n', case=False, na=False)]
        data = data[data.str.contains('e', case=False, na=False)]
        return json.dumps(data)
    
    @property
    def filter_df_endinz(self):
        # read & prep
        data = pd.read_csv(self.file, header=None)
        data = data.iloc[:, 0]
        # filter
        data = data[data.str.endswith('z')|data.str.endswith('Z')]
        return json.dumps(data)
    
    @property
    def filter_df_no_vowels(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None)
        data = data.iloc[:, 0]
        # filter
        data = data[data.str.count(r'[aeiouAEIOU]') == 0]
        data = data.values.tolist()
        return json.dumps(data)
    
    def __str__(self):
        return self.title

    @property
    def user_username(self):
        self.user.username
        return (self.user.username)
@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)
