import json

class Command(object):

  def __init__(self, type, params = {}):
    self.type   = type
    self.params = params

  # http://stackoverflow.com/a/18647629/1185698
  def _jsonSupport(*args):
    def default(self, xObject):
      return {
        'type':   xObject.type,
        'params': xObject.params,
      }

    json.JSONEncoder.default = default
    json._default_decoder    = json.JSONDecoder()

  _jsonSupport()
