#http://pbpython.com/plotly-dash-intro.html
#https://dash.plot.ly/getting-started
#Dash - A New Framework for Building User Interfaces for Technical Computing | SciPy 2017 | Chris Par
#https://www.youtube.com/watch?v=sea2K4AuPOk
#https://stackoverflow.com/questions/48939151/python-dash-without-using-external-urls-for-js

#sudo -H pip3 install dash==0.18.3
#sudo -H pip3 install dash-renderer==0.10.0
#sudo -H pip3 install dash-html-components==0.7.0
#sudo -H pip3 install dash-core-components==0.12.6
#sudo -H pip3 install plotly --upgrade

import dash
import dash_core_components
import dash_html_components

graph_data = [] 
graph_data += [{'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'}]
graph_data += [{'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Montreal'}]
graph = dash_core_components.Graph(id='example-graph', figure={'data':graph_data, 'layout':{'title':'A visualization made by dask'}})

app = dash.Dash()
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.layout = dash_html_components.Div(children=[
                       dash_html_components.H1(children='Hello Dash'),
                       dash_html_components.Div(children='''
                       hello there. wrinting this from dash
                       '''),
                       graph])

if __name__ == '__main__':
  app.run_server(debug=True)

