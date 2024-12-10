#Source: https://stackoverflow.com/questions/53308766/bokehgetting-typeerror-object-of-type-polygon-is-not-json-serializable
# <------- This is where the graph starts------->

# reset the graph
reset_output()

# import data
data = gpd.read_file("/Users/xxxx/Desktop/cb_2015_us_state_500k/cb_2015_us_state_500k.shp", encoding="utf-8")
data1 = data[~data.STUSPS.isin(['AK','AS', 'GU', 'HI', 'PR','MP', 'VI'])]
data2 = data[data.STUSPS.isin(['TX', 'UT'])]

# get a list of unique value
unique_state = sorted(list(data2.NAME.unique()))
select = Select(title="State", options=unique_state)

# get data into ColumnDataSource
source=ColumnDataSource(ColumnDataSource.from_df(data2.loc[:]))

# crate filtered dataframe
filteredSource = ColumnDataSource(data=dict(STUSPS=[],NAME=[],ALAND=[]))

columns = [TableColumn(field="NAME",title="NAME",sortable=True),
           TableColumn(field="STUSPS",title="STUSPS",sortable=True),
           TableColumn(field="ALAND",title="ALAND",sortable=True),
           TableColumn(field="geometry",title="geometry",sortable=True)]

data_table=DataTable(source=filteredSource,columns=columns, width=800 )

# <---- Call back starts ---->
callback = CustomJS(args=dict(source=source,
                              filteredSource=filteredSource,
                              data_table=data_table), code="""
var data = source.data;
var f = cb_obj.value;
var d2 = filteredSource.data;
d2['STUSPS']=[]
d2['NAME']=[]
d2['ALAND']=[]
d2['geometry']=[]


for(i = 0; i < data['NAME'].length;i++){

if(data['NAME'][i]==f){

    d2['STUSPS'].push(data['STUSPS'][i])
    d2['NAME'].push(data['NAME'][i])
    d2['ALAND'].push(data['ALAND'][i])
    d2['geometry'].push(data['geometry'][i])
}

}

filteredSource.change.emit()
// trigger change on datatable
data_table.change.emit()

""")

select.js_on_change('value',callback)

layout = column(widgetbox(select, data_table))

# output_file("filter.html", title="filter example")

show(layout)