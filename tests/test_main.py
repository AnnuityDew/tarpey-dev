# import native Python packages

# import third party packages
from fastapi.testclient import TestClient

# import custom local stuff
from src.main import create_fastapi_app
from src.tarpeydev.mildredleague import MLSeason
from src.tarpeydev.timecapsule import TemplateID, DeepTemplateID

# create test app and client
app = create_fastapi_app()
client = TestClient(app)


# autobracket negative distribution test
# games don't go into negative time test
# tests that ensure independent events don't happen in the same possession

def test_all_simple_views():
    '''Test all simple views (one-template public GET views with no arguments).'''
    tagged_url_dicts = [
        {'tags': route.__dict__.get('tags'), 'path': route.path, 'method': route.methods}
        for route in app.routes
        if route.__dict__.get('tags')
    ]
    
    # filter to the simple view tag
    test_routes = [
        url_dict for url_dict in tagged_url_dicts
        if 'simple_view' in url_dict['tags']
    ]

    # get request on each view
    response_list = [
        client.get(url_dict.get('path')) for url_dict in test_routes
    ]

    # make sure all responses are 200!
    assert all(response.status_code == 200 for response in response_list)


def test_all_enumerated_path_views():
    '''Test all enumerated path views (public GET views with no arguments).'''
    tagged_url_dicts = [
        {
            'tags': route.__dict__.get('tags'),
            'path': route.path,
            'method': route.methods,
            'parameters': list(route.__dict__.get('param_convertors').keys()),
        }
        for route in app.routes
        if route.__dict__.get('tags')
    ]

    # filter to the enumerated path view tag
    test_routes = [
        url_dict for url_dict in tagged_url_dicts
        if 'enumerated_path_view' in url_dict['tags']
    ]

    # create lookups of path and variable to enumeration class name
    path_lookups = {
        (test_routes[0]['path'], test_routes[0]['parameters'][0]): [member.value for name, member in MLSeason.__members__.items()],
        (test_routes[1]['path'], test_routes[1]['parameters'][0]): [member.value for name, member in TemplateID.__members__.items()],
        (test_routes[2]['path'], test_routes[2]['parameters'][0]): [member.value for name, member in DeepTemplateID.__members__.items()],
    }

    # use format to create list of all paths
    expanded_test_routes = []
    print(path_lookups)
    for path, value_array in path_lookups.items():
        for value in value_array:
            expanded_test_routes.append(path[0].format(**{path[1]: value}))

    # get request on each view
    response_list = [
        client.get(url) for url in expanded_test_routes
    ]

    # make sure all responses are 200!
    assert all(response.status_code == 200 for response in response_list)


def test_all_form_post_views():
    '''Test all form POST views.'''
    tagged_url_dicts = [
        {
            'tags': route.__dict__.get('tags'),
            'path': route.path,
            'method': route.methods,
            'parameters': list(route.__dict__.get('param_convertors').keys()),
        }
        for route in app.routes
        if route.__dict__.get('tags')
    ]

    # filter to the enumerated path view tag
    test_routes = [
        url_dict for url_dict in tagged_url_dicts
        if 'form_post_view' in url_dict['tags']
    ]
    assert not test_routes


def test_all_query_views():
    '''Test all query views.'''
    tagged_url_dicts = [
        {
            'tags': route.__dict__.get('tags'),
            'path': route.path,
            'method': route.methods,
            'parameters': list(route.__dict__.get('param_convertors').keys()),
        }
        for route in app.routes
        if route.__dict__.get('tags')
    ]

    # filter to the enumerated path view tag
    test_routes = [
        url_dict for url_dict in tagged_url_dicts
        if 'query_view' in url_dict['tags']
    ]
    assert not test_routes


def test_for_untagged_views():
    '''Look for untagged views that aren't being tested!'''
    tagged_url_dicts = [
        {
            'tags': route.__dict__.get('tags'),
            'path': route.path,
            'parameters': list(route.__dict__.get('param_convertors').keys()),
        }
        for route in app.routes
        if not route.__dict__.get('tags')
    ]
    print(tagged_url_dicts)
    assert not tagged_url_dicts


def test_api_routes():
    '''For this one, we should separate out the API app creation into a separate file.
    
    This will be helpful anyway once we totally disconnect frontend...
    '''
    assert False
