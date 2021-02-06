Live at https://tarpey.dev/
Dev version live at https://dev.tarpey.dev (unless I broke something)

References:

FastAPI — How to add basic and cookie authentication
https://medium.com/data-rebels/fastapi-how-to-add-basic-and-cookie-authentication-a45c85ef47d3

https with url_for
https://github.com/encode/starlette/issues/538

Good JWT articles
https://hasura.io/blog/best-practices-of-using-jwt-with-graphql/
https://stormpath.com/blog/where-to-store-your-jwts-cookies-vs-html5-web-storage

Accessing request.state directly in starlette:
https://stackoverflow.com/questions/63273028/fastapi-get-user-id-from-api-key

Colors!
https://refactoringui.com/previews/building-your-color-palette/

Docker file system
https://stackoverflow.com/questions/20813486/exploring-docker-containers-file-system

Best practices for testing and packaging
https://docs.pytest.org/en/latest/goodpractices.html#install-package-with-pip
https://blog.ionelmc.ro/2014/05/25/python-packaging/

Async fetch
https://stackoverflow.com/questions/46241827/fetch-api-requesting-multiple-get-requests
https://stackoverflow.com/questions/54685210/calling-sync-functions-from-async-function

MongoDB with FastAPI
https://github.com/tiangolo/fastapi/issues/1515
https://github.com/tiangolo/fastapi/issues/452

multiprocessing
https://stackoverflow.com/questions/63169865/how-to-do-multiprocessing-in-fastapi
https://github.com/tiangolo/fastapi/issues/1487#issuecomment-657290725

There's so much that can go wrong when you're building one of these things.

* One wrong / can lead to a redirect that costs you your cookies. =[
* 'Authorization: Bearer' might not be exactly what your backend calls it.
