# Theory

## Message-passing Graph Convolution Networks

## Charge prediction with the charge equilibration method


:::{raw} html

<style>
:root {
    --arrow-thickness: 1px;
    --arrow-head-size: 10px;
    --arrow-color: black;
}
.arrow.thick {
    --arrow-thickness: 3px;
    --arrow-head-size: 10px;
}
.arrow::after {
    width: var(--arrow-head-size);
    height: var(--arrow-head-size);
    visibility: visible;
    content: "";
    margin-left: calc(-1 * var(--arrow-head-size));;
    vertical-align: middle;
    border: solid var(--arrow-color);
    border-width: 0 var(--arrow-thickness) var(--arrow-thickness) 0;
    display: inline-block;
    padding: var(--arrow-thickness);
    transform: rotate(-45deg);
}
.arrow::before {
    content: attr(label);
    background-color: var(--arrow-color);
    width: 100%;
    height: var(--arrow-thickness);
    overflow-y: visible;
    display: inline-block;
    vertical-align: middle;
    font-size: 0.7em;
}
.arrow {
    display: inline-block;
    flex-basis: 15px;
    flex-grow: 1;
}

.flowchart {
    display: flex;
    align-items: center;
    text-align: center;
}
.flowchart * {
    margin: 5px;
}
.flowchart em {
    font-style: normal;
    font-weight: bold;
}
.flowchart.topdown {
    flex-direction: column;
}

.module {
    border-radius: 12px;
    padding: 12px;
    display: flex;
    align-items: center;
    flex-grow: 1;
    align-self: stretch;
    position: relative;
}
.module::before {
    content: attr(label);
    font-size: 0.7em;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
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
    <div class="module blue" label="Convolution">
        <div> Message-passing</div>
        <div class="arrow"></div>
        <div> Activation </div>
    </div>
    <div class="arrow"></div>
    <div class="module orange" label="Readout">Neural net</div>
    <div class="arrow thick"></div>
    <div><em>Prediction</em></div>
</div>

:::