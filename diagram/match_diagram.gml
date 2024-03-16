graph [
  multigraph 1
  node [
    id 0
    label "f_right_z[10]"
    type 0
    phase "[1.5]"
  ]
  node [
    id 1
    label "f_right_x[11]"
    type 2
    phase "[1.5]"
  ]
  node [
    id 2
    label "f_right_z[12]"
    type 0
    phase "[1.5]"
  ]
  node [
    id 3
    label "f_right_x[13]"
    type 2
    phase "[0.5]"
  ]
  node [
    id 4
    label "f_right_z[14]"
    type 0
    phase "[0.5]"
  ]
  node [
    id 5
    label "f_right_x[15]"
    type 2
    phase "[1.0]"
  ]
  node [
    id 6
    label "f_right_z[16]"
    type 0
    phase "[0.0]"
  ]
  node [
    id 7
    label "f_right_x[17]"
    type 2
    phase "[0.75]"
  ]
  node [
    id 8
    label "f_right_z[18]"
    type 0
    phase "[0.0]"
  ]
  node [
    id 9
    label "f_right_x[19]"
    type 2
    phase "[1.5]"
  ]
  node [
    id 10
    label "f_left_z[12, 18]"
    type 1
    phase "[-1]"
  ]
  node [
    id 11
    label "f_left_x[15, 17]"
    type 3
    phase "[-1]"
  ]
  node [
    id 12
    label "f_left_x[17, 19]"
    type 3
    phase "[-1]"
  ]
  edge [
    source 0
    target 5
    key 0
    type 1
  ]
  edge [
    source 1
    target 4
    key 0
    type 1
  ]
  edge [
    source 2
    target 8
    key 0
    type 1
  ]
  edge [
    source 2
    target 10
    key 0
    type 0
  ]
  edge [
    source 3
    target 6
    key 0
    type 1
  ]
  edge [
    source 5
    target 7
    key 0
    type 1
  ]
  edge [
    source 5
    target 11
    key 0
    type 0
  ]
  edge [
    source 6
    target 7
    key 0
    type 1
  ]
  edge [
    source 7
    target 9
    key 0
    type 1
  ]
  edge [
    source 7
    target 11
    key 0
    type 0
  ]
  edge [
    source 7
    target 12
    key 0
    type 0
  ]
  edge [
    source 8
    target 9
    key 0
    type 1
  ]
  edge [
    source 8
    target 10
    key 0
    type 0
  ]
  edge [
    source 9
    target 12
    key 0
    type 0
  ]
]
