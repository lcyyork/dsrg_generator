from src.Index import Index
from src.Term import Term
from src.Tensor import Tensor
from src.SQOperator import SecondQuantizedOperator


make_tensor = Tensor.make_tensor
make_sq = SecondQuantizedOperator


def test_init():
    indices_type = 'spin-orbital'
    list_of_tensors = [Tensor.make_tensor('Hamiltonian', "g0,g1,c0", "g2,p0,v0", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "p0,p1,g3", "a0,h1,a1", indices_type),
                       Tensor.make_tensor('Kronecker', "v0", "p1", indices_type),
                       Tensor.make_tensor('cumulant', "h1", "c0", indices_type),
                       Tensor.make_tensor('cumulant', "g2,a0", "g0,g1", indices_type),
                       Tensor.make_tensor('cumulant', "a1", "g3", indices_type)]
    sq_op = SecondQuantizedOperator.make_empty(indices_type)
    a = Term(list_of_tensors, sq_op)
    print(a)
    # print(a.next_index_number)
    # print(a._downgrade_cumulant_indices())
    # print(a)
    # print(a.next_index_number)
    # a._remove_kronecker_delta()
    # print(a)
    # print(a.next_index_number)
    a.simplify(simplify_core_cumulant=True)
    print(a)
    print(a.next_index_number)

    # for row in a.build_adjacency_matrix():
    #     print(row)
    # for row in a.order_tensors():
    #     print(row)
    print(a)
    print(a.canonicalize_sympy())

    list_of_tensors = [Tensor.make_tensor('Hamiltonian', "v0,c0", "v1,c1", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "v1", "c2", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "v2,v3", "c0,c3", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "c2,c3", "v0,v3", indices_type)]
    sq_op = SecondQuantizedOperator("c1", "v2", indices_type)

    a = Term(list_of_tensors, sq_op)
    # print(a.canonicalize())

    print(a)
    print(a.canonicalize_sympy())

    # for i in a.canonicalize().generate_spin_cases_naive():
    #     print(i)


def test_init_2():
    a = Term([make_tensor('H', 'g0', 'g0')], make_sq("g0", "g0"))
    print(a)

    b = Term.from_term(a, True)
    print(b)


def test_perm_part():
    list_of_tensors = [make_tensor('H', 'g0, g1', 'g2, p0'), make_tensor('T', 'h0, h1', 'p0, p1')]
    sq_op = make_sq('g0, g1, p1', 'g2, h0, h1')
    a = Term(list_of_tensors, sq_op)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('g0'), Index('g1')], [Index('p1')]]
    assert ann_part == [[Index('g2')], [Index('h0'), Index('h1')]]

    list_of_tensors = [make_tensor('H', 'v4, c2', 'v3, c1'),
                       make_tensor('T', 'c1', 'v0'),
                       make_tensor('T', 'c0, c3', 'v1, v4'),
                       make_tensor('T', 'c2, c3', 'v2, v3')]
    sq_op = make_sq('v2, c0', 'v0, v1')
    a = Term(list_of_tensors, sq_op)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('c0')], [Index('v2')]]
    assert ann_part == [[Index('v0')], [Index('v1')]]

    list_of_tensors = [make_tensor('H', 'v3, v4', 'v0, c3'),
                       make_tensor('T', 'c0, c1', 'v1, v3'),
                       make_tensor('T', 'c2, c3', 'v2, v4')]
    sq_op = make_sq('v0, v1, v2', 'c0, c1, c2')
    a = Term(list_of_tensors, sq_op)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('v0')], [Index('v1')], [Index('v2')]]
    assert ann_part == [[Index('c0'), Index('c1')], [Index('c2')]]

    list_of_tensors = [make_tensor('H', 'g0, h1', 'p1, g0')]
    sq_op = make_sq('g0, p1', 'g0, h1')
    a = Term(list_of_tensors, sq_op, -1)
    cre_part, ann_part = a.perm_partition_open()
    assert cre_part == [[Index('g0')], [Index('p1')]]
    assert ann_part == [[Index('g0')], [Index('h1')]]


def test_problem():
    """
    -1/2 & H^{ v_{0} }_{ v_{1} } T^{ v_{1} }_{ c_{1} } T^{ v_{2} }_{ c_{0} } T^{ c_{0} c_{1} }_{ v_{0} v_{2} }
    -1/2 & H^{ v_{0} }_{ v_{1} } T^{ c_{0} }_{ v_{2} } T^{ c_{1} }_{ v_{0} } T^{ v_{1} v_{2} }_{ c_{0} c_{1} }
    """
    indices_type = 'spin-orbital'
    list_of_tensors = [Tensor.make_tensor('Hamiltonian', "v0", "v1", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "v1", "c1", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "v2", "c0", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "c0,c1", "v0,v2", indices_type)]
    sq_op = SecondQuantizedOperator.make_empty(indices_type)
    a = Term(list_of_tensors, sq_op)

    list_of_tensors = [Tensor.make_tensor('Hamiltonian', "v0", "v1", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "c0", "v2", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "c1", "v0", indices_type),
                       Tensor.make_tensor('cluster_amplitude', "v1,v2", "c0,c1", indices_type)]
    b = Term(list_of_tensors, sq_op)

    ac = a.canonicalize_sympy()
    bc = b.canonicalize_sympy()
    print(ac)
    print(bc)
    print(ac == bc)
