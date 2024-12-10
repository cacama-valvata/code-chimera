from . import db
from flask_restful import Resource, reqparse, abort, fields, marshal_with
from .models import TodoModel
from flask import jsonify

# used to retrive passed in arguments
_Todo_post_args = reqparse.RequestParser()
_Todo_post_args.add_argument('content', type=str, help='To do task', required = True)

# define how an object should be serialized. Return using these fields (paired with @marshal_with)
_todo_resource_fields = {
    'id': fields.Integer,
    'content': fields.String
}

class TodoListResource(Resource):
    def get(self):
        #https://stackoverflow.com/questions/7102754/jsonify-a-sqlalchemy-result-set-in-flask
        tasks = TodoModel.query.order_by(TodoModel.date_created).all()
        json_list=[i.serialize for i in tasks]
        return jsonify(json_list)
    #should also handle adding (post) to list

class TodoResource(Resource):
    #handles getting(get), updating(put), & deleting (del)  individual tasks
    @marshal_with(_todo_resource_fields)
    def get(self, task_id):
        task = TodoModel.query.filter_by(id=task_id).first()
        if not task:
            #returns <Response 404> and the .json will be message
            abort(404, message = "Could not find task with that ID")
        return task, 200

    @marshal_with(_todo_resource_fields)
    def post(self, task_id = None):
        args = _Todo_post_args.parse_args()
        #if need to check unique abort(409, message = "__ taken")
        Todo = TodoModel(content = args['content'])
        db.session.add(Todo)
        db.session.commit()
        return Todo, 201

    @marshal_with(_todo_resource_fields)
    def delete(self, task_id):
        task = TodoModel.query.filter_by(id=task_id).first()

        #Alterantiive query method
        #task  = TodoModel.query.get_or_404(id)
        if not task:
            abort(404, message = "Could not find task with that ID")
        db.session.delete(task)
        db.session.commit()
        return {'Note': 'Deleted'}, 204
        
    @marshal_with(_todo_resource_fields)
    def put(self, task_id):
        task = TodoModel.query.filter_by(id=task_id).first()
        if not task:
            abort(404, message = "Could not find task with that ID")
        args = _Todo_post_args.parse_args()
        for arg in args:
            setattr(task, arg, args[arg])

        db.session.commit()
        return task, 201
