"""
Author: M.A. Marosvolgyi 2021
(c)Anywi
"""

from flask import Flask, render_template, redirect, request, url_for
from markupsafe import Markup
import prototype


"""
    got help from:
        https://stackoverflow.com/questions/32019733/getting-value-from-select-tag-using-flask
        https://stackoverflow.com/questions/12502646/access-multiselect-form-field-in-flask
        https://stackoverflow.com/questions/30323224/deploying-a-minimal-flask-app-in-docker-server-connection-issues

"""

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', data = data, guides = guides)

@app.route("/test" , methods=['GET', 'POST'])
def test():
    select = request.form.getlist('comp_select')
    listOfGuides = processor.getOccurencesOfOrganisation(select)
    return render_template('index.html', data = data, guides = listOfGuides) # just to see what select is

@app.route('/table')
def table():
    theMatrix = Markup(prototype.dumpMatrix(processor))
    return render_template('table.html', theMatrix = theMatrix)


if __name__=='__main__':
    print ("Setup server")

    gl = prototype.CodeOfPracticeReader()
    gl.setup()
    guides = gl.getGuides()
    processor = prototype.Processor(guides)

    myList = list(processor.generateSetOfOrganisations())

    data = []

    for i in myList:
        data.append({'name':i})



    app.run(host='0.0.0.0')
