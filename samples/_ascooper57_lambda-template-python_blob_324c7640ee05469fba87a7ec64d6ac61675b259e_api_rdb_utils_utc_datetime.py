# -*- coding: utf-8 -*-

import datetime as dt
import re

from dateutil.tz import tzoffset

iso_format = '%Y-%m-%dT%H:%M:%S+00:00'
iso8601_format = '%Y-%m-%dT%H:%M:%S.%f'


# https://gist.github.com/squioc/3078803
def epoch_to_iso8601(timestamp):
    """
    epoch_to_iso8601 - convert the unix epoch time into a iso8601 formatted date
    >>> epoch_to_iso8601(1341872870)
    '2012-07-09T22:27:50.2725172'
    """
    return dt.datetime.fromtimestamp(timestamp).isoformat()


def iso8601_to_epoch(datestring):
    # type: (str) -> int
    """
    iso8601_to_epoch - convert the iso8601 date into the unix epoch time
    >>> iso8601_to_epoch('2012-07-09T22:27:50.272517')
    1341872870
    """
    global iso8601_format
    datetime_obj = dt.datetime.strptime(datestring, iso8601_format).replace(tzinfo=dt.timezone.utc)
    return datetime_to_epoch(datetime_obj)


def iso_to_epoch(datestring):
    # type: (str) -> int
    """
    iso8601_to_epoch - convert the iso8601 date into the unix epoch time
    >>> iso8601_to_epoch('2012-07-09T22:27:50+00:00')
    1341872870
    """
    global iso_format
    datetime_obj = dt.datetime.strptime(datestring, iso_format).replace(tzinfo=dt.timezone.utc)
    return datetime_to_epoch(datetime_obj)


# https://stackoverflow.com/questions/27804342/how-do-i-filter-and-extract-raw-log-event-data-from-amazon-cloudwatch
def datetime_to_epoch(datetime_obj):
    # type: (object) -> int
    from calendar import timegm
    # noinspection PyUnresolvedReferences
    return timegm(datetime_obj.utctimetuple()) * 1000


def json_to_dt(obj):
    if obj.pop('__type__', None) != "datetime":
        return obj
    # zone, offset = obj.pop("tz")
    obj.pop("tz")
    obj["tzinfo"] = tzoffset("UTC", 0)
    datetime_obj = dt.datetime(**obj)
    return datetime_obj.isoformat()


def dt_to_json(obj):
    if isinstance(obj, dt.datetime):
        return {
            "__type__": "datetime",
            "year": obj.year,
            "month": obj.month,
            "day": obj.day,
            "hour": obj.hour,
            "minute": obj.minute,
            "second": obj.second,
            "microsecond": obj.microsecond,
            "tz": (obj.tzinfo.tzname(obj), obj.utcoffset().total_seconds())
        }
    else:
        raise TypeError("Can't serialize {}".format(obj))


def as_seconds(secs):
    """
    Return a duration string as seconds
    """
    if isinstance(secs, (int, float,)):
        return int(secs)

    # exact seconds
    m = re.match(r'^\d+$', secs)
    if m:
        return int(secs)

    # minutes
    m = re.match(r'^(\d+)m(?!on)(in)?', secs, re.I)
    if m:
        return int(int(m.group(1)) * 60)

    # hours
    m = re.match(r'^(\d+)h', secs, re.I)
    if m:
        return int(m.group(1)) * 3600

    # days
    m = re.match(r'^(\d+)d', secs, re.I)
    if m:
        return int(m.group(1)) * 86400

    # weeks
    m = re.match(r'^(\d+)w', secs, re.I)
    if m:
        return int(m.group(1)) * 86400 * 7

    # months
    m = re.match(r'^(\d+)mon', secs, re.I)
    if m:
        return int(int(m.group(1)) * 86400 * 365 / 12)

    # years
    m = re.match(r'^(\d+)y', secs, re.I)
    if m:
        return int(m.group(1)) * 86400 * 365

    raise ValueError(secs)
