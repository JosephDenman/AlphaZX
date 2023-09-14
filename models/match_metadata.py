from diagram.pyzx_nx_conv import Z_NTYPE_NAME, X_NTYPE_NAME

f_left_z_metadata = ([Z_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', Z_NTYPE_NAME)])

f_left_x_metadata = ([X_NTYPE_NAME], [(X_NTYPE_NAME, 'to', X_NTYPE_NAME)])

f_right_z_metadata = ([Z_NTYPE_NAME, X_NTYPE_NAME],
                      [(Z_NTYPE_NAME, 'to', Z_NTYPE_NAME), (Z_NTYPE_NAME, 'to', X_NTYPE_NAME),
                       (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)])

f_right_x_metadata = ([Z_NTYPE_NAME, X_NTYPE_NAME],
                      [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME),
                       (X_NTYPE_NAME, 'to', X_NTYPE_NAME)])

b_left_metadata = (
    [Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)])

b_right_metadata = (
    [Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)])

y_right_z_metadata = (
    [Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)])

y_right_x_metadata = (
    [Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)])

y_left_z_metadata = (
    [Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)])

y_left_x_metadata = (
    [Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)])

match_metadata_dict = {
    'f_left_z': ([Z_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', Z_NTYPE_NAME)]),
    'f_left_x': ([X_NTYPE_NAME], [(X_NTYPE_NAME, 'to', X_NTYPE_NAME)]),
    'f_right_z': ([Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', Z_NTYPE_NAME), (Z_NTYPE_NAME, 'to', X_NTYPE_NAME),
                                                 (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)]),
    'f_right_x': ([Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME),
                                                 (X_NTYPE_NAME, 'to', X_NTYPE_NAME)]),
    'b_left': ([Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)]),
    'b_right': ([Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)]),
    'y_right_z': (
    [Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)]),
    'y_right_x': (
    [Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)]),
    'y_left_z': (
    [Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)]),
    'y_left_x': ([Z_NTYPE_NAME, X_NTYPE_NAME], [(Z_NTYPE_NAME, 'to', X_NTYPE_NAME), (X_NTYPE_NAME, 'to', Z_NTYPE_NAME)])
}
