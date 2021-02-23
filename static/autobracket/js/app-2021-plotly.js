const fetchData = async () => {
    let allData = await Promise.all([
        fetch("/api/autobracket/sim/2020/UCONN/GA/margins").then(data => data.json()),
    ]);
    return allData;
}

async function pageRender() {
    // grab the data
    const allData = await fetchData();
    const histData = allData[0];
    var data = [
        {
            x: histData,
            type: 'histogram',
            xbins: {
                start: Math.min(histData) - 0.5,
                size: 1,
                end: Math.max(histData) + 0.5,
            },
        }
    ]
    var layout = {
        template: tarpeydevDefault,
        title: "Margin",
    };
    var config = {
        responsive: true,
    };
    Plotly.newPlot('analysis', data, layout, config);
}

window.onload = pageRender;
