import json, boto3
from data import Data
from .service import Service
class S3(Service):

  def __init__(self, region, bucket='backup', path='/'):
    self.region = region
    self.bucket = bucket
    self.path = path
    # s3 = boto3.resource("s3").Bucket("bucket")
    # json.load_s3 = lambda f: json.load(s3.Object(key=f).get()["Body"])
    # json.dump_s3 = lambda obj, f: s3.Object(key=f).put(Body=json.dumps(obj))


  def connect(self, connection={'aws_access_key_id':None, 'aws_secret_access_key':None}):
    # see https://stackoverflow.com/questions/40336918/how-to-write-a-file-or-data-to-an-s3-object-using-boto3
    self.s3 = boto3.resource(
      's3',
      region_name=self.region,
      aws_access_key_id=connection['aws_access_key_id'],
      aws_secret_access_key=connection['aws_secret_access_key']
    )

  def read(self, job: Job) -> Data:
    data = Data()
    obj = s3.Object(self.bucket, self.path+job.from_time+'.jsonl')
    content = obj.get()['Body'].read()
    for line in content.splitlines():
      doc_data = json.loads(line)
      data.add_doc(doc_data)
    return data

  def write(self, data: Data, job: Job) -> Data:
    docs = data.get_docs()
    content=[]
    for doc in docs:
      l = json.dumps(doc, sort_keys=True, default=str)
      content.append(l)
    self.s3.Object(self.bucket, self.path+job.from_time+'.jsonl').put(Body='\n'.join(content))
    return data