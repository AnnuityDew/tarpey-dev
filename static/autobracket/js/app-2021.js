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
    const histData = allData[0]['hist_data']

    // chart sizing and labeling stuff
    const width = 1000;
    const height = 1000;
    const margin = ({top: 20, right: 120, bottom: 30, left: 120})
    const padding = 28;
    var chartHeader = document.getElementById("exhibit-title")

    function minutesHist() {
        bins = d3.bin().thresholds(40)(histData)

        x = d3.scaleLinear()
            .domain([bins[0].x0, bins[bins.length - 1].x1])
            .range([margin.left, width - margin.right])
        y = d3.scaleLinear()
            .domain([0, d3.max(bins, d => d.length)]).nice()
            .range([height - margin.bottom, margin.top])
        
        const svg = d3.select("#analysis")
            .append("svg")
            .attr("id", "current-chart")
            .attr("preserveAspectRatio", "xMinYMin meet")
            .attr("viewBox", [0, 0, width, height])
            .style("opacity", 0);

        svg.append("g")
            .attr("fill", "#7f2626")
            .selectAll("rect")
            .data(bins)
            .join("rect")
            .attr("x", d => x(d.x0) + 1)
            .attr("width", d => Math.max(0, x(d.x1) - x(d.x0) - 1))
            .attr("y", d => y(d.length))
            .attr("height", d => y(0) - y(d.length));

        // x-axis
        svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`)
            .attr("class", "axis-ticks")
            .call(d3.axisBottom(x).ticks(width / 80).tickSizeOuter(0))
            .call(g => g.append("text")
                .attr("x", width - margin.right)
                .attr("y", -4)
                .attr("fill", "currentColor")
                .attr("text-anchor", "end")
                .text(histData.x))
        // y-axis
        svg.append("g").attr("transform", `translate(${margin.left},0)`)
            .attr("class", "axis-ticks")
            .call(d3.axisLeft(y).ticks(height / 40))
            .call(g => g.select(".domain").remove())
            .call(g => g.select(".tick:last-of-type text").clone()
                .attr("x", 4)
                .attr("text-anchor", "start")
                .text(histData.y))

        d3.select("#current-chart")
            .transition()
            .duration(1000)
            .style("opacity", 1)
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
            .attr("class", "axis-ticks")
            .each(function (d) { return d3.select(this).call(xAxis.scale(d)); })
            .call(g => g.select(".domain").remove())
            .call(g => g.selectAll(".tick line").attr("stroke", "#444444"));

        svg.append("g")
            .selectAll("g").data(y).join("g")
            .attr("transform", (d, i) => `translate(0,${i * size})`)
            .attr("class", "axis-ticks")
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
    var currentPage = 0;

    // list of pages
    const pageList = ["scatter", "histogram - 2020 minutes"];
    chartHeader.textContent = pageList[currentPage];

    function nextPage() {
        switch (pageList[currentPage]) {
            case pageList[0]:
                d3Transition();
                // call the next page
                setTimeout(minutesHist, (1000));
                currentPage++;
                break;
            case pageList[1]:
                d3Transition();
                // call the next page
                setTimeout(scatterChart, (1000));
                currentPage++;
                break;
            default:
                break;
        };
        // if user was on the last page, go back to the first page
        // and vice versa
        if (currentPage >= pageList.length) {
            currentPage = 0
        };
        // rename the page
        chartHeader.textContent = pageList[currentPage];
    }

    function lastPage() {
        switch (pageList[currentPage]) {
            case pageList[0]:
                d3Transition();
                // call the next page
                setTimeout(minutesHist, (2000));
                currentPage--;
                break;
            case pageList[1]:
                d3Transition();
                // call the next page
                setTimeout(scatterChart, (2000));
                currentPage--;
                break;
            default:
                break;
        };
        // if user was on the first page, go back to the last page
        // and vice versa
        if (currentPage < 0) {
            currentPage = pageList.length - 1
        };
        // rename the page
        chartHeader.textContent = pageList[currentPage];
    }

    function d3Transition() {
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
    }

    function d3ToPlotly() {
        // this function isn't working perfectly, but it's a good start.
        // fade current exhibit out (100% transparent), then remove
        d3.select("#current-chart")
            .transition()
            .duration(1000)
            .style("opacity", 0);
        // without this timeout the chart will instantly vanish
        setTimeout(function () {
            d3.select("#current-chart")
                .remove();
            // remove the d3-specific class
            var container = document.getElementById("analysis");
            container.classList.remove("svg-container");
        }, (1000));

    }

    function d3FromPlotly() {
        // this function isn't working perfectly, but it's a good start.
        // fade current exhibit out (100% transparent)
        Plotly.purge("analysis");
        // remove the Plotly class and add the d3-specific class
        var container = document.getElementById("analysis");
        container.classList.remove("js-plotly-plot");
        container.classList.add("svg-container");
    }

    // add click functionality
    var backButton = document.getElementById("back-button")
    var nextButton = document.getElementById("next-button")
    backButton.addEventListener("click", lastPage);
    nextButton.addEventListener("click", nextPage);
}

window.onload = pageRender;
