from src.diagram.diagram_generators import clifford_zx_match_diagram

d = clifford_zx_match_diagram(2, 2, True)
d.to_pyg_hdata()
print(d)