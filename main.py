# from flask import escape
# import json
import asyncio
from usecase import summarize_usecase


def getSummarization(request):
    request_json = request.get_json(silent=True)
    try:
        result = asyncio.run(summarize_usecase.getSummaryResult(request_json['text'], request_json['useClustering']))
        return {"status": 200, "message": "success", "result": result}
    except Exception as e:
        return {"status": 500, "error": "Summarize Error", "message": e, }


def hello_world(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    print(request_json)
    print(request_args)

    if request_json and 'name' in request_json:
        name = request_json['name']
    elif request_args and 'name' in request_args:
        name = request_args['name']
    else:
        name = 'World'

    return {
        "name": name,
        "type": "Test",
        "status": 200
    }
