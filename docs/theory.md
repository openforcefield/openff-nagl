# Theory

Non-polarizable biomolecular force fields like those produced by OpenFF need fixed partial charges for all their atoms to model long-range non-bonded interactions. Usually, these charges are computed via quantum chemical calculations, but this approach has a few drawbacks:

1. Quantum chemical calculations model conformation-dependent polarization effects, but the same charges will be used for all molecular mechanics configurations
2. QC calculations are computationally expensive, especially for larger molecules
3. QC calculations are complex, require specialized software and domain knowledge, and can be difficult to execute, especially when accounting for the above drawbacks

Most force fields work around this issue by pre-computing and distributing libraries of charges for their target molecules. OpenFF force fields are designed to support a very wide variety of molecules, including those unknown to the developers, so this approach doesn't work. As a result, the slowest computational step in the parametrization of an OpenFF topology is computing partial charges, even when proprietary software is used. For large and unusual molecules like proteins with non-canonical amino acids, neither approach is really workable.

NAGL solves this problem by training machine learning models to predict partial charges - and potentially other properties - from a **molecular graph**. Machine learning describes a wide array of techniques that can produce  a cheap-to-compute approximation to an arbitrary mathematical function by interpolating between example evaluations of the function in a high-dimensional space. This addresses all three drawbacks:

1. Because the input graph is based on the connectivity of the atoms rather than their co-ordinates, the output is conformation-independent (as long as the training data was properly prepared)
2. The model can be evaluated much more quickly than a QC calculation, and scales to large molecules efficiently
3. NAGL can take advantage of the thriving software ecosystem around machine learning, and so can be executed on all sorts of molecules and machines with relative ease

In trade, machine learning models require large datasets of examples of the desired property. As they are at heart interpolations between these examples, the details of the structure of the models and the design of the example datasets are crucial to the production of reliable models. NAGL is designed to make this possible.

## The Molecular Graph

A molecular graph is essentially a Lewis structure represented as a mathematical graph. A  graph is a collection of **nodes** that are connected by **edges**. Graphs can model lots of everyday systems and objects: in computer networks, computers are nodes and WiFi or Ethernet connections are edges; for an airline, airports are nodes and flights are edges; on Twitter, accounts are nodes and follows are edges. 

In a molecular graph, atoms are nodes and bonds are edges. A molecular graph can store extra information in its nodes and edges; in a Lewis structure, this is things like an atom's element and lone pairs, or a bond's bond order or stereochemistry, but for a molecular graph they can be anything. Element and bond order are usually somewhere near the top of the list. Envisioning a molecule as a graph lets us apply computer science techniques designed for graphs to molecules.

Note that a particular molecular species may have more than one molecular graph topology that represents it. This most commonly happens with tautomers and resonance forms. NAGL operates on molecular graphs, and on the OpenFF [`Molecule`] class, and it doesn't try to be clever about whether two molecules are the same or not. If you want tautomers to produce the same charges, you will need to prepare your dataset and featurization accordingly.

:::{figure-md} fig-diff_graphs
![Neutral and zwitterionic alanine](_static/images/theory/alanine_zwitterion.svg)

Neutral and zwitterionic alanine. These might be considered the same molecular species, but they are certainly different molecular graphs.
:::

[`Molecule`]: openff.toolkit.topology.Molecule

## Featurization

Before we can put a molecule through a neural network, we need a way to represent it with numerical vectors. NAGL uses the molecular graph to break the molecule into nodes and edges, and then represents each node or edge as its own vector. The numbers in these vectors each describe a particular feature of the atom or bond, and so the vectors are called **feature vectors**. The process of encoding an atom or bond as a feature vector is called **featurization**.

When a feature is naturally represented as a continuous numerical value, and the learned function depends on that value in predictable ways, the value may be used as the feature directly. This is generally not the case for molecular graphs. For example, while the element of an atom could be represented by its atomic number, the model would then have to encode some understanding of the periodic table to interpret their chemistry. Instead, the model generally benefits from being able to consider elements as different categories, especially when they are only trained on a subset of the elements. 

NAGL therefore uses **one-hot encoding** for most featurization, in which each feature $f$ represents a property that the atom or bond either has ($f=1$) or lacks ($f=0$). The featurization is therefore like a list of boolean values. For example, the first four features might specify the element of an atom:

- `[1 0 0 0]` represents Hydrogen
- `[0 1 0 0]` represents Carbon
- `[0 0 1 0]` represents Nitrogen
- `[0 0 0 1]` represents Oxygen

This prevents the model from assuming that adjacent elements are similar. Note that these values are represented internally as floating point numbers, not single bits. The model's internal representation is therefore free to mix and scale them as needed, which allows the model to represent a carbon atom (#6) with some oxygen character (#8) without the result appearing like nitrogen (#7).

## The Convolution Module: Message-passing Graph Convolutional Networks

NAGL's goal is to produce machine-learned models that can compute partial charges and other properties for all the atoms in a molecule. To do this, it needs a way to represent atoms to the network in their full molecular context. This representation is called an **embedding**, because it embeds all the information there is to know about a particular atom in some relatively low-dimensional space. An atom's feature vector is a simplistic, human-readable embedding, but we want something that a neural network can use to infer charges, even if that means losing simplicity and readability. That means folding in information about the surrounding atoms and their connectivity.

NAGL produces atom embeddings with a message-passing graph convolutional network (GCN). A GCN takes each node's feature vector and iteratively mixes it with those of progressively more distant neighbours to produce an embedding for the node. This embedding can then be passed on to a **readout** network that predicts some particular property of interest. Both the **convolution module** and **readout module** can be trained in concert to produce an embedding that is bespoke to the computed property. 

https://docs.dgl.ai/en/1.0.x/tutorials/models/1_gnn/1_gcn.html

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
    --flowchart-spacing: 10px;
    --label-size: 0.8em;
    --bg-color: white;
    --fg-color: black;
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
    margin: calc(0.2071 * var(--arrow-head-size));
    border: solid var(--fg-color);
    border-width: 0 var(--arrow-thickness) var(--arrow-thickness) 0;
    display: inline-block;
    transform: rotate(-45deg);
    position: absolute;
    right: 0;
    top: var(--arrow-thickness);
    z-index: -1;
}
.arrow::before {
    content: "";
    border-bottom: var(--fg-color) solid var(--arrow-thickness);
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
    line-height: 1.2;
    padding: 0 var(--arrow-head-size);
    flex: 1 1 0px;
    font-size: var(--label-size);
    position: relative;
    height: calc(
        var(--arrow-thickness) 
        + 2 * var(--arrow-head-size)
    );
    text-decoration: underline var(--bg-color) 1rem;
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
    margin: 0 var(--flowchart-spacing);
    line-height: 1.8;
}
.arrow.fullwidth::after {
    transform: rotate(45deg);
    background-image: linear-gradient(
        45deg,
        transparent calc(50% - var(--arrow-thickness)/2), 
        var(--fg-color) calc(50% - var(--arrow-thickness)/2), 
        var(--fg-color) calc(50% + var(--arrow-thickness)/2), 
        transparent calc(50% + var(--arrow-thickness)/2)
    );
    margin: calc(0.2071 * var(--arrow-head-size));
    left: var(--arrow-thickness);
    top: calc(2 * var(--arrow-head-size));
}
.arrow.fullwidth::before {
    border-right: var(--fg-color) solid var(--arrow-thickness);
    width: calc(100% - 2 * var(--arrow-head-size) - 2 * var(--flowchart-spacing));
    height: calc(2 * var(--arrow-head-size));
    top:0;
    left: var(--arrow-head-size);
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
    left: var(--arrow-thickness);
    top: 0;
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
    container-type: inline-size;
    container-name: flowchart;
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
    border: solid 1px var(--fg-color);
    z-index: -1;
    color: var(--fg-color);
}

.flowchart .module {
    display: flex;
    align-items: center;
    align-content: center;
    position: relative;
    gap: var(--flowchart-spacing);
    border: none;
    flex-wrap: wrap;
    background: var(--bg-color);
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

.flowchart .module.blue {
    --fg-color: white;
    --bg-color: #2f9ed2;
}
.flowchart .module.orange {
    --fg-color: white;
    --bg-color: #f03a21;
}

.flowchart > div:not(.arrow):not(.module) > *:first-child {  
    margin: 0 auto var(--flowchart-spacing) auto;
    text-align: left;
    max-width: fit-content;
}

@container flowchart (max-width: 550px) {
    .flowchart > * {
        flex-basis: 100%;
    }
    .arrow, .arrow.fullwidth {
        height: unset;
        width: unset;
        margin: 0 auto;
        line-height: 1.8;
        padding: calc(2 * var(--arrow-head-size)) 0
    }
    .arrow::after, .arrow.fullwidth::after {
        transform: rotate(45deg);
        background: none;
        top: calc(100% - 2*var(--arrow-head-size) + var(--arrow-thickness));
        left: calc(50% - var(--arrow-head-size));
    }
    .arrow::before, .arrow.fullwidth::before {
        border: none;
        width: var(--arrow-thickness);
        background-color: var(--fg-color);
        height: 100%;
        top: 0;
        left: calc(50% - var(--arrow-thickness)/2);
    }
    
    .arrow.fullwidth.loopback {
        position: absolute;
        right: var(--flowchart-spacing);
        top: calc(var(--flowchart-spacing) + var(--label-size));
        height: calc(100% - 2*var(--flowchart-spacing) - var(--label-size) - 2*var(--arrow-head-size));
        margin: 0;
        padding: var(--arrow-head-size) 0;
        max-width: 10%;
        line-height: 1.2;
        text-decoration-thickness: 1.2rem;
        writing-mode: vertical-rl;
        text-orientation: mixed;
    }
    
    .arrow.fullwidth.loopback::before {
        background: none;
        border: var(--fg-color) solid var(--arrow-thickness);
        border-left: none;
        height: calc(100% - 2*var(--arrow-head-size));
        width: calc(2*var(--arrow-head-size));
        left: -50%;
        top: var(--arrow-head-size);
    }
    
    .arrow.fullwidth.loopback::after {
        right: calc(100% - var(--arrow-head-size)/2);
        top: calc(var(--arrow-thickness)/2);
        transform: rotate(135deg)
    }
    
    .flowchart .module {
        flex-wrap: nowrap;
        flex-direction: column;
        position: relative;
    }

}

</style>
<div class="flowchart">
    <div>
        <div>Molecule</div>
        <img class="block" src="_static/images/theory/alanine.svg">
    </div>
    <div class="arrow">featurize</div>
    <div>
        <ul>
            <li>Feature vectors </li>
            <li>Adjacency matrix</li>
        </ul>
        <img src="_static/images/theory/alanine-atom-features.svg">
        <img src="_static/images/theory/alanine-graph.svg">
    </div>
    <div class="arrow fullwidth"></div>
    <div class="module blue" label="Convolution module">
        <div><img src="_static/images/theory/alanine-message_passing_input.svg"></div>
        <div class="arrow">Message-passing</div>
        <div><img src="_static/images/theory/alanine-message_passing_output.svg"></div>
        <div class="arrow">Update</div>
        <div><img src="_static/images/theory/alanine-update_output.svg"></div>
        <div class="arrow fullwidth loopback">Iterate with greater hop distance</div>
    </div>
    <div class="arrow fullwidth"></div>
    <div class="module orange" label="Readout module">
        <div>Neural net</div>
        <div class="arrow"></div>
        <div>Post-processing</div>
    </div>
    <div class="arrow thick"></div>
    <div><em>Prediction</em></div>
</div>

:::

## Charge prediction with the charge equilibration method