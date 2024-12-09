from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.auth.models import User
import pandas as pd
import json


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
    def filter_all(self):
        # read & prep
        data = pd.read_csv(self.file.url, header=None)
        data = data.iloc[:, 0]
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
    
    