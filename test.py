import theano
from theano import tensor as T
from theano import function
import numpy as np

a = np.array([[[1, 2, 3], [3, 4, 5]], [[7, 8, 9], [45, 345, 12]]])

x0 = T.ftensor3()
def create_atom_context(atom_vector):
    # type_vector = T.fvector()
    types_array = atom_vector[0]
    dists = atom_vector[1]
    w = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]

    # outputs_info = T.as_tensor_variable(np.asarray(0, dtype=np.float32))
    # types, updates = theano.scan(fn=lambda atm_type: atm_type,
    #                              outputs_info=None,
    #                              sequences=type_vector)
    # mult = type_vector*2
    # f = function(inputs=[type_vector], outputs=mult)
    # print(f([1, 2, 3]))

    # f = function(inputs=[type_vector], outputs=types)
    # return T.concatenate([types_array], [dists])
    print(len(types_array))
    ls = []
    for tp in types_array:
        ls.append(tp)
    return ls
                                                

rows, updates = theano.scan(fn=create_atom_context,
                            outputs_info=None,
                            sequences=x0)

matrix = T.stacklists(rows)

f = function(inputs=[x0], outputs=matrix)

test_matrix = np.asarray([[[1, 1, 1, 1], [2, 2, 2, 2]], [[3, 3, 3, 3], [4, 4, 4, 4]]], dtype=np.float32)
print(f(test_matrix))
