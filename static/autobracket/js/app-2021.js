const fetchData = async () => {
    let allData = await Promise.all([
        fetch("/api/autobracket/sim/2020/kmeans").then(data => data.json()),
    ]);
    return allData;
}

async function pageRender() {
    // grab the data
    const allData = await fetchData();
    const scatterData = allData[0]['scatter_data']
    const scatterColumns = allData[0]['scatter_columns']

    // page width and padding
    const padding = 28
    const width = 954

    function minutesHist() {
        var xAxis = d3.attr("transform", `translate(0,${height - margin.bottom})`)
            .call(d3.axisBottom(x).ticks(width / 80).tickSizeOuter(0))
            .call(g => g.append("text")
                .attr("x", width - margin.right)
                .attr("y", -4)
                .attr("fill", "currentColor")
                .attr("font-weight", "bold")
                .attr("text-anchor", "end")
                .text(data.x))

        var yAxis = d3.attr("transform", `translate(${margin.left},0)`)
            .call(d3.axisLeft(y).ticks(height / 40))
            .call(g => g.select(".domain").remove())
            .call(g => g.select(".tick:last-of-type text").clone()
                .attr("x", 4)
                .attr("text-anchor", "start")
                .attr("font-weight", "bold")
                .text(data.y))
    }

    function scatterChart() {
        const size = (width - (scatterColumns.length + 1) * padding) / scatterColumns.length + padding

        const svg = d3.select("#analysis")
            .append("svg")
            .attr("id", "current-chart")
            .attr("preserveAspectRatio", "xMinYMin meet")
            .attr("viewBox", [-padding, 0, width, width])
            .style("opacity", 0)

        svg.append("style")
            .text(`circle.hidden { fill: #000; fill-opacity: 1; r: 1px; }`);

        var x = scatterColumns.map(c => d3.scaleLinear()
            .domain(d3.extent(scatterData, d => d[c]))
            .rangeRound([padding / 2, size - padding / 2]))

        var y = x.map(x => x.copy().range([size - padding / 2, padding / 2]))

        var z = d3.scaleOrdinal()
            .domain(scatterData.map(d => d.player_type))
            .range(d3.schemeCategory10)

        var xAxis = d3.axisBottom()
            .ticks(6)
            .tickSize(size * scatterColumns.length);

        var yAxis = d3.axisLeft()
            .ticks(6)
            .tickSize(-size * scatterColumns.length);

        svg.append("g")
            .selectAll("g").data(x).join("g")
            .attr("transform", (d, i) => `translate(${i * size},0)`)
            .each(function (d) { return d3.select(this).call(xAxis.scale(d)); })
            .call(g => g.select(".domain").remove())
            .call(g => g.selectAll(".tick line").attr("stroke", "#444444"));

        svg.append("g")
            .selectAll("g").data(y).join("g")
            .attr("transform", (d, i) => `translate(0,${i * size})`)
            .each(function (d) { return d3.select(this).call(yAxis.scale(d)); })
            .call(g => g.select(".domain").remove())
            .call(g => g.selectAll(".tick line").attr("stroke", "#444444"));

        function brush(cell, circle, svg) {
            const brush = d3.brush()
                .extent([[padding / 2, padding / 2], [size - padding / 2, size - padding / 2]])
                .on("start", brushstarted)
                .on("brush", brushed)
                .on("end", brushended);

            cell.call(brush);

            let brushCell;

            // Clear the previously-active brush, if any.
            function brushstarted() {
                if (brushCell !== this) {
                    d3.select(brushCell).call(brush.move, null);
                    brushCell = this;
                }
            }

            // Highlight the selected circles.
            function brushed({ selection }, [i, j]) {
                let selected = [];
                if (selection) {
                    const [[x0, y0], [x1, y1]] = selection;
                    circle.classed("hidden",
                        d => x0 > x[i](d[scatterColumns[i]])
                            || x1 < x[i](d[scatterColumns[i]])
                            || y0 > y[j](d[scatterColumns[j]])
                            || y1 < y[j](d[scatterColumns[j]]));
                    selected = scatterData.filter(
                        d => x0 < x[i](d[scatterColumns[i]])
                            && x1 > x[i](d[scatterColumns[i]])
                            && y0 < y[j](d[scatterColumns[j]])
                            && y1 > y[j](d[scatterColumns[j]]));
                }
                svg.property("value", selected).dispatch("input");
            }

            // If the brush is empty, select all circles.
            function brushended({ selection }) {
                if (selection) return;
                svg.property("value", []).dispatch("input");
                circle.classed("hidden", false);
            }
        }

        const cell = svg.append("g")
            .selectAll("g")
            .data(d3.cross(d3.range(scatterColumns.length), d3.range(scatterColumns.length)))
            .join("g")
            .attr("transform", ([i, j]) => `translate(${i * size},${j * size})`);

        cell.append("rect")
            .attr("fill", "none")
            .attr("stroke", "#bbbbbb")
            .attr("x", padding / 2 + 0.5)
            .attr("y", padding / 2 + 0.5)
            .attr("width", size - padding)
            .attr("height", size - padding);

        cell.each(function ([i, j]) {
            d3.select(this).selectAll("circle")
                .data(scatterData.filter(d => !isNaN(d[scatterColumns[i]]) && !isNaN(d[scatterColumns[j]])))
                .join("circle")
                .attr("cx", d => x[i](d[scatterColumns[i]]))
                .attr("cy", d => y[j](d[scatterColumns[j]]));
        });

        const circle = cell.selectAll("circle")
            .attr("r", 3.5)
            .attr("fill-opacity", 0.7)
            .attr("fill", d => z(d.player_type));

        cell.call(brush, circle, svg);

        svg.append("g")
            .style("font", "bold 16px sans-serif")
            .style("pointer-events", "none")
            .style("fill", "white")
            .selectAll("text")
            .data(scatterColumns)
            .join("text")
            .attr("transform", (d, i) => `translate(${i * size},${i * size})`)
            .attr("x", padding)
            .attr("y", padding)
            .attr("dy", ".71em")
            .text(d => d);

        svg.property("value", []);

        d3.select("#current-chart")
            .transition()
            .duration(1000)
            .style("opacity", 1)
    }

    // render the initial chart
    scatterChart();

    function nextPage() {
        // fade current exhibit out (100% transparent), then remove
        d3.select("#current-chart")
            .transition()
            .duration(1000)
            .style("opacity", 0);

        // without this timeout the chart will instantly vanish
        setTimeout(function () {
            d3.select("#current-chart")
                .remove();
        }, (1000));

        // call the next page
        setTimeout(scatterChart, (1000));
    }

    // add click functionality
    var backButton = document.getElementById("back-button")
    var nextButton = document.getElementById("next-button")
    backButton.addEventListener("click", nextPage);
    nextButton.addEventListener("click", nextPage);

    // something like this for figuring out what page we're on...
    // https://stackoverflow.com/questions/256754/how-to-pass-arguments-to-addeventlistener-listener-function
}

window.onload = pageRender;
