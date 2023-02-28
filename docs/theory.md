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
    --arrow-pos-offset: 0;
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
    width: 100%;
    display: inline-block;
    font-size: 0.7em;
    position: absolute;
    left: 0;
    top: 50%;
    z-index: -1;
}
.arrow {
    display: inline-block;
    min-width: calc(2 * var(--arrow-head-size));
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
    --arrow-head-size: 10px;
    flex-basis: 100%;
    height: calc(
        var(--arrow-thickness) 
        + 4 * var(--arrow-head-size)
    );
    font-size: 1.2em;
}
.arrow.fullwidth::after {
    transform: rotate(45deg);
    float: left;
    margin-left: 0;
    background-image: linear-gradient(
        45deg,
        transparent calc(50% - var(--arrow-thickness)/2), 
        black calc(50% - var(--arrow-thickness)/2), 
        black calc(50% + var(--arrow-thickness)/2), 
        transparent calc(50% + var(--arrow-thickness)/2)
    );
    position:absolute;
    left: calc(var(--arrow-head-pos-offset) + var(--arrow-thickness));
    top: calc(var(--arrow-head-pos-offset) + 2 * var(--arrow-head-size));
    z-index: -10;
}
.arrow.fullwidth::before {
    border-right: var(--arrow-color) solid var(--arrow-thickness);
    width: calc(100% - 2 * var(--arrow-head-size));
    height: calc(2 * var(--arrow-head-size));
    position: absolute;
    top:0;
    left: var(--arrow-head-size);
    z-index: -10;
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
    z-index: -10;
}

.flowchart .module {
    --arrow-head-size: 5px;
    --flowchart-spacing: 7px;
    display: flex;
    align-items: center;
    position: relative;
    gap: var(--flowchart-spacing);
    border: none;
}
.flowchart .module[label] {
    padding-top: 15px;
}
.flowchart .module::before {
    content: attr(label);
    font-size: 0.7em;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
}
.flowchart .module > * {
    margin-top: var(--flowchart-spacing);
    margin-bottom: var(--flowchart-spacing);
}

.flowchart .module.blue {
    background: #2f9ed2;
    color: white;
    --arrow-color: white;
}
.flowchart .module.orange {
    background: #f03a21;
    color: white;
    --arrow-color: white;
}
</style>
<div class="flowchart">
    <div>
    Molecule
    </div>
    <div class="arrow">Label</div>
    <div class="module blue" label="Message-passing convolution">
        <div>Aggregate</div>
        <div class="arrow"></div>
        <div>Update</div>
    </div>
    <div class="arrow"></div>
    <div class="module orange" label="Readout">Neural net</div>
    <div class="arrow thick fullwidth">Testing</div>
    <div><em>Prediction</em></div>
</div>

:::

## Charge prediction with the charge equilibration method