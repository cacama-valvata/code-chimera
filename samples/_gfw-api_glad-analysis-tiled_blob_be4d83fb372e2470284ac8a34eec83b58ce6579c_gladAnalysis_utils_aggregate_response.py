import datetime
from collections import defaultdict


def format_alerts_geom(rows, request):

    agg_by = request.args.get('aggregate_by')

    # pull first row from sql response [(<alert_date>, <alert_count>), ()]
    if rows:
        first_alert_count = rows[0][1]

    # if we're aggregating values with no results, rows looks like []
    else:
        first_alert_count = None

    # if we have at least one (date, alert_count) row,
    # and aggregating, build day/month/year dict
    if first_alert_count and agg_by:
        return rows_to_agg_dict(rows, agg_by)

    # not aggregating, only one value to return
    elif first_alert_count:
        return first_alert_count

    # if aggregating but no alerts
    elif agg_by:
        return []

    # if not aggregating and no alerts
    else:
        return 0


def rows_to_agg_dict(rows, agg_by):

    # convert rows [(<date_str), count), (<date_str, count), ...]
    # to {date_obj: count} dict
    date_dict = {datetime.datetime.strptime(k, '%Y-%m-%d'): v
                for (k, v) in rows}
    k = date_dict.keys() # alert date = datetime.datetime(2015, 6, 4, 0, 0)
    v = date_dict.values() # count

    resp_dict = {
                 'year': grouped_and_to_rows([x.year for x in k], v, 'year'),
                 # month --> quarter calc: https://stackoverflow.com/questions/1406131
                 'quarter': grouped_and_to_rows([(x.year, (x.month-1)//3 + 1) for x in k], v, 'quarter'),
                 'month':  grouped_and_to_rows([(x.year, x.month) for x in k], v, 'month'),
                 'week': grouped_and_to_rows([(x.year, x.isocalendar()[1]) for x in k], v, 'week'),
                 'day': grouped_and_to_rows([(x.year, x.strftime('%Y-%m-%d')) for x in k], v, 'day'),
                 'total': sum(v)
                }

    return resp_dict[agg_by]


def grouped_and_to_rows(keys, vals, agg_type):

    # source: https://jakevdp.github.io/blog/2017/03/22/group-by-from-scratch/
    count = defaultdict(int)
    for key, val in zip(keys, vals):
        count[key] += val
    grouped = dict(count)

    final_list = []

    for key, val in grouped.iteritems():

        if agg_type == 'year':
            row = {agg_type: key}
        else:
            row = {'year': key[0], agg_type: key[1]}

        row['count'] = val
        final_list.append(row)

    return final_list
