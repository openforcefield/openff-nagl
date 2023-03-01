# Theory



## Message-passing Graph Convolution Networks

https://tkipf.github.io/graph-convolutional-networks/

http://www.aritrasen.com/graph-neural-network-message-passing-gcn-1-1/

- Nodes of input graph are featurized
- Message-passing convolution to generate graph's embedding
    + Features of neighbors are mixed in to each node
    + Iterate to capture long-range effects
- Readout processes convolved embedding to make prediction

:::{raw} html

<style>
:root {
    --arrow-thickness: 1.5px;
    --arrow-head-size: 7px;
    --arrow-color: black;
    --arrow-head-pos-offset: 0;
    --flowchart-spacing: 10px;
    --label-size: 0.8em;
}
.arrow.thick {
    --arrow-thickness: 4px;
    --arrow-head-size: 10px;
}
.arrow::after {
    width: calc(1.4142 * var(--arrow-head-size));
    height: calc(1.4142 * var(--arrow-head-size));
    content: "";
    padding: 0;
    margin: 0;
    border: solid var(--arrow-color);
    border-width: 0 var(--arrow-thickness) var(--arrow-thickness) 0;
    display: inline-block;
    transform: rotate(-45deg);
    --arrow-head-pos-offset: calc(0.2071 * var(--arrow-head-size));
    position: absolute;
    right: var(--arrow-head-pos-offset);
    top: calc(var(--arrow-thickness) + var(--arrow-head-pos-offset));
    z-index: -1;
}
.arrow::before {
    content: "";
    border-bottom: var(--arrow-color) solid var(--arrow-thickness);
    height: 0;
    width: calc(100% - var(--arrow-thickness));
    display: inline-block;
    position: absolute;
    left: 0;
    top: calc(50% - var(--arrow-thickness)/2);
    z-index: -1;
}
.arrow {
    display: inline-block;
    padding: 0 var(--arrow-head-size);
    flex-grow: 1;
    flex-shrink: 0;
    font-size: 0.7em;
    position: relative;
    height: calc(
        var(--arrow-thickness) 
        + 2 * var(--arrow-head-size)
    );
    text-decoration: underline white 1rem;
    text-decoration-skip-ink: none;
    text-underline-position: under;
    text-underline-offset: -1rem;
}

.arrow.fullwidth {
    flex-basis: 100%;
    height: calc(
        var(--arrow-thickness) 
        + 4 * var(--arrow-head-size)
    );
    font-size: 1.2em;
    margin: 0 var(--flowchart-spacing);
}
.arrow.fullwidth::after {
    transform: rotate(45deg);
    background-image: linear-gradient(
        45deg,
        transparent calc(50% - var(--arrow-thickness)/2), 
        var(--arrow-color) calc(50% - var(--arrow-thickness)/2), 
        var(--arrow-color) calc(50% + var(--arrow-thickness)/2), 
        transparent calc(50% + var(--arrow-thickness)/2)
    );
    position:absolute;
    left: calc(var(--arrow-head-pos-offset) + var(--arrow-thickness));
    top: calc(var(--arrow-head-pos-offset) + 2 * var(--arrow-head-size));
    z-index: -1;
}
.arrow.fullwidth::before {
    border-right: var(--arrow-color) solid var(--arrow-thickness);
    width: calc(100% - 2 * var(--arrow-head-size) - 2 * var(--flowchart-spacing));
    height: calc(2 * var(--arrow-head-size));
    position: absolute;
    top:0;
    left: var(--arrow-head-size);
    z-index: -1;
}

.arrow.fullwidth.loopback {
    height: calc(
        var(--arrow-thickness) 
        + 2 * var(--arrow-head-size)
    );
}
.arrow.fullwidth.loopback::after {
    transform: rotate(-135deg);
    position:absolute;
    left: calc(var(--arrow-head-pos-offset) + var(--arrow-thickness));
    top: var(--arrow-head-pos-offset);
    z-index: -1;
}

.flowchart {
    display: flex;
    align-items: center;
    text-align: center;
    gap: var(--flowchart-spacing);
    padding: var(--flowchart-spacing) 0;
    flex-wrap: wrap;
    max-width: 100%;
}
.flowchart em {
    font-style: normal;
    font-weight: bold;
}
.flowchart.topdown {
    flex-direction: column;
}

.flowchart > *:not(.arrow) {
    flex-grow: 1;
    border-radius: 12px;
    padding: 12px;
    align-self: stretch;
    border: solid 1px black;
    z-index: -1;
}

.flowchart .module {
    display: flex;
    align-items: center;
    align-content: center;
    position: relative;
    gap: var(--flowchart-spacing);
    border: none;
    flex-wrap: wrap;
}
.flowchart .module[label] {
    padding-top: calc(var(--label-size) + var(--flowchart-spacing));
}
.flowchart .module::before {
    content: attr(label);
    font-size: var(--label-size);
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    font-weight: bold;
}

.flowchart .module.blue, .flowchart .module.orange {
    color: white;
    --arrow-color: white;
}
.flowchart .module.blue, .flowchart .module.orange {
    background: #2f9ed2;
}
.flowchart .module.orange {
    background: #f03a21;
}

.flowchart > div > *:first-child {  
    margin: 0 auto var(--flowchart-spacing) auto;
    text-align: left;
    max-width: fit-content;
}

</style>
<div class="flowchart">
    <div>
        <div>Molecule</div>
        <img class="block" src="_static/images/theory/alanine.svg">
    </div>
    <div class="arrow">featurization</div>
    <div>
        <ul>
            <li>Feature vectors </li>
            <li>Adjacency matrix</li>
        </ul>
        <img src="_static/images/theory/alanine-atom-features.svg">
        <img src="_static/images/theory/alanine-graph.svg">
    </div>
    <div class="arrow fullwidth"></div>
    <div class="module blue" label="Message-passing convolution">
        <div>Aggregate</div>
        <div class="arrow"></div>
        <div>Activate</div>
        <div class="arrow"></div>
        <div>Update</div>
        <div class="arrow fullwidth loopback"></div>
    </div>
    <div class="arrow fullwidth"></div>
    <div class="module orange" label="Readout">
        <div>Neural net</div>
        <div class="arrow"></div>
        <div>Post-processing</div>
    </div>
    <div class="arrow thick"></div>
    <div><em>Prediction</em></div>
</div>

:::

## Charge prediction with the charge equilibration method