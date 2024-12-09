"""
Author: M.A. Marosvolgyi 2021
(c)Anywi
"""

from flask import Flask, render_template, redirect, request, url_for
import prototype


"""
    got help from:
        https://stackoverflow.com/questions/32019733/getting-value-from-select-tag-using-flask
        https://stackoverflow.com/questions/12502646/access-multiselect-form-field-in-flask

"""

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', data = data)

@app.route("/test" , methods=['GET', 'POST'])
def test():
    select = request.form.getlist('comp_select')
    listOfGuides = processor.getOccurencesOfOrganisation(select)
    return render_template('result.html', data = listOfGuides) # just to see what select is

if __name__=='__main__':

    print ("Setup server")

    gl = prototype.CodeOfPracticeReader()
    gl.setup()
    guides = gl.getGuides()
    processor = prototype.Processor(guides)

    list = list(processor.generateSetOfOrganisations())

    data = []
    for i in list:
        data.append({'name':i})

    app.run(debug=True)