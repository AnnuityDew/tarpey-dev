Live at https://tarpey.dev/
Dev version live at https://dev.tarpey.dev (unless I broke something)

The stack from front-to-back:
HTML (where it all began)
CSS (custom + components from Shoelace)
JavaScript (for charts, D3 and Plotly)
FastAPI in Python for the backend
MongoDB (interaction using Motor and ODMantic)
Hosted on Google Cloud Run

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

JavaScript events
https://stackoverflow.com/questions/256754/how-to-pass-arguments-to-addeventlistener-listener-function
https://stackoverflow.com/questions/45353852/load-data-from-api-onclick

D3
https://chartio.com/resources/tutorials/how-to-resize-an-svg-when-the-window-is-resized-in-d3-js/
https://www.d3-graph-gallery.com/graph/interactivity_transition.html

MongoDB with FastAPI
https://github.com/tiangolo/fastapi/issues/1515
https://github.com/tiangolo/fastapi/issues/452

multiprocessing
https://stackoverflow.com/questions/63169865/how-to-do-multiprocessing-in-fastapi
https://github.com/tiangolo/fastapi/issues/1487#issuecomment-657290725
https://www.cloudcity.io/blog/2019/02/27/things-i-wish-they-told-me-about-multiprocessing-in-python/
https://stackoverflow.com/questions/20387510/proper-way-to-use-multiprocessor-pool-in-a-nested-loop
https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join

we'll never stop learning pandas
https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe

licenses
For autobracket, specifically the scatter plot:
https://observablehq.com/@d3/brushable-scatterplot-matrix
Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

There's so much that can go wrong when you're building one of these things.

* One wrong / can lead to a redirect that costs you your cookies. =[
* 'Authorization: Bearer' might not be exactly what your backend calls it.
