import tldextract

url = "https://stackoverflow.com/users/1076426/dsafa"
ext = tldextract.extract(url)
print(ext)
print(ext.subdomain)
print('.'.join(ext[:3]))