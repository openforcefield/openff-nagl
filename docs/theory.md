# Theory



## Message-passing Graph Convolution Networks

https://tkipf.github.io/graph-convolutional-networks/

- Nodes of input graph are featurized
- Message-passing convolution to generate graph's embedding
    + Features of neighbors are mixed in to each node
    + Iterate to capture long-range effects
- Readout processes convolved embedding to make prediction

:::{raw} html

<style>
:root {
    --arrow-thickness: 1px;
    --arrow-head-size: 7px;
    --arrow-color: black;
    --arrow-label-position: 1.5em;
    --flowchart-spacing: 10px;
}
.arrow.thick {
    --arrow-thickness: 3px;
    --arrow-head-size: 10px;
}
.arrow::after {
    width: calc(1.4142 * var(--arrow-head-size));
    height: calc(1.4142 * var(--arrow-head-size));
    content: "";
    padding: 0;
    margin: 0;
    margin-left: calc(-1.5 * var(--arrow-head-size));
    vertical-align: middle;
    border: solid var(--arrow-color);
    border-width: 0 var(--arrow-thickness) var(--arrow-thickness) 0;
    display: inline-block;
    transform: rotate(-45deg);
}
.arrow::before {
    content: attr(label);
    border-bottom: var(--arrow-color) solid var(--arrow-thickness);
    height: 0;
    width: 100%;
    overflow-y: visible;
    display: inline-block;
    vertical-align: middle;
    font-size: 0.7em;
}
.arrow[label]::before {
    padding: 0 min(calc(var(--arrow-head-size) / 2 + 5px), calc(5em - var(--arrow-head-size)));
    padding-bottom: var(--arrow-label-position);
    margin-top: calc(-1 * var(--arrow-label-position));
}
.arrow {
    display: inline-block;
    min-width: calc(2 * var(--arrow-head-size));
    flex-grow: 1;
    flex-shrink: 0;
}

.flowchart {
    display: flex;
    align-items: center;
    text-align: center;
    gap: var(--flowchart-spacing);
    padding: var(--flowchart-spacing) 0;
    overflow-x: auto;
}
.flowchart em {
    font-style: normal;
    font-weight: bold;
}
.flowchart.topdown {
    flex-direction: column;
}

.module {
    --arrow-head-size: 5px;
    --flowchart-spacing: 7px;
    border-radius: 12px;
    padding: 12px;
    display: flex;
    align-items: center;
    flex-grow: 1;
    align-self: stretch;
    position: relative;
    gap: var(--flowchart-spacing);
}
.module[label] {
    padding-top: 15px;
}
.module::before {
    content: attr(label);
    font-size: 0.7em;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
}
.module > * {
    margin-top: var(--flowchart-spacing);
    margin-bottom: var(--flowchart-spacing);
}

.module.blue {
    background: #2f9ed2;
    color: white;
    --arrow-color: white;
}
.module.orange {
    background: #f03a21;
    color: white;
    --arrow-color: white;
}
</style>
<div class="flowchart">
    <div>
    Molecule
    </div>
    <div class="arrow" label="featurization"></div>
    <div class="module blue" label="Message-passing convolution">
        <div>Aggregate</div>
        <div class="arrow"></div>
        <div>Update</div>
    </div>
    <div class="arrow"></div>
    <div class="module orange" label="Readout">Neural net</div>
    <div class="arrow thick"></div>
    <div><em>Prediction</em></div>
</div>

:::

## Charge prediction with the charge equilibration method