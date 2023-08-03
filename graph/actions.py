"""
Taken from page 589 of PQP

Vertex = int

------------------------------

Phase = float

NewEdges = int

OldEdges = List[Boolean]

Spider Fission {
    vertex: Vertex
    phase: Phase
    newEdges: NewEdges
    oldEdges: OldEdges
}

pre-conditions (spiderFission):
    1. spiderFission.vertex exists
    2. spiderFission.vertex is a spider
    3. spiderFission.newEdges is <= max_new_edges and >= 1
    4. len(spiderFission.oldEdges) matches len(E(spiderFission.vertex)) in G

post-conditions:

------------------------------

SpiderFusion {
    u: Vertex
    v: Vertex
}

pre-conditions (spiderFusion):
    1. spiderFusion.u and spiderFusion.v exist in G
    2. G.is_spider(spiderFusion.u) and G.is_spider(spiderFusion.v)
    3. G.basis(u) is G.basis(v)

post-conditions:

------------------------------

ColorChange = { v: Vertex }

pre-conditions (colorChange):
    1. G.is_spider(colorChange.v)
    2. G.phase(colorChange.v) is 0
    3. G.num_neighbors(colorChange.v) is 3
    4. if G.basis(colorChange.v) is Z:
            1. for_all(G.neighbors(colorChange.v), lambda u: G.basis(u) is X)
            2. has_correct_phases(G.neighbors(colorChange.v))
       elif G.basis(colorChange.v) is X:
            1. for_all(G.neighbors(colorChange.v), lambda u: G.basis(u) is Z)
            2. has_correct_phases(G.neighbors(colorChange.v))

post-conditions:

------------------------------

BiAlgebra = {
    u: Vertex
    v: Vertex
    w: Vertex
    x: Vertex
}

pre-conditions (bi_algebra):
    1.

------------------------------
Matching

Assuming there is a way to generate random graphs using only generators from pg 589:

1. If there are no fission actions available, then there are no spiders
2. If there are no fusion actions available
3. Bi-algebra occurrences must be matched explicitly, action retrieval.

-----

post-conditions (step):
    1. if there are no nodes besides boundary nodes in the initial observation, the game completes, no reward is given.
    2. if there are no nodes besides boundary nodes left after applying a rule, the game completes, the
       normal reward is given.
    3. there should never be an isolated graph
    4.

"""
