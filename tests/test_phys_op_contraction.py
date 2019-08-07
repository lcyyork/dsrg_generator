from collections import defaultdict
from timeit import default_timer as timer
from dsrg_generator.Term import Term, hamiltonian_operator, cluster_operator
from dsrg_generator.Tensor import Tensor
from dsrg_generator.SQOperator import SecondQuantizedOperator
from dsrg_generator.phys_op_contraction import contract_terms, combine_terms
from dsrg_generator.phys_op_contraction import single_commutator, recursive_single_commutator
from dsrg_generator.phys_op_contraction import bch_cc_rsc, nested_commutator_cc
from dsrg_generator.phys_op_contraction import print_terms_ambit_functions, save_terms_ambit_functions


make_tensor = Tensor.make_tensor
make_sq = SecondQuantizedOperator


def test_contract_terms_1():
    h = hamiltonian_operator(2)
    t2e = cluster_operator(2, hole_label='c', particle_label='v')
    t2ee = cluster_operator(2, start=4, hole_label='c', particle_label='v')

    a = contract_terms([h, t2e, t2ee], max_cu=1, scale_factor=0.5, for_commutator=True, n_process=2)

    ref = {Term([make_tensor('H', 'c2,c3', 'v2,v3'),
                 make_tensor('t', 'c0,c1', 'v0,v2'), make_tensor('t', 'c2,c3', 'v1,v3')],
                make_sq('v0,v1', 'c0,c1'), -0.25),
           Term([make_tensor('H', 'c2,c3', 'v2,v3'),
                 make_tensor('t', 'c0,c1', 'v2,v3'), make_tensor('t', 'c2,c3', 'v0,v1')],
                make_sq('v0,v1', 'c0,c1'), 1.0/16.0),
           Term([make_tensor('H', 'c2,c3', 'v2,v3'),
                 make_tensor('t', 'c0,c2', 'v0,v1'), make_tensor('t', 'c1,c3', 'v2,v3')],
                make_sq('v0,v1', 'c0,c1'), -0.25),
           Term([make_tensor('H', 'c2,c3', 'v2,v3'),
                 make_tensor('t', 'c0,c2', 'v0,v2'), make_tensor('t', 'c1,c3', 'v1,v3')],
                make_sq('v0,v1', 'c0,c1'), 0.5),
           Term([make_tensor('H', 'v2,v3', 'g0,c3'),
                 make_tensor('t', 'c0,c1', 'v0,v2'), make_tensor('t', 'c2,c3', 'v1,v3')],
                make_sq('g0,v0,v1', 'c0,c1,c2'), -0.5),
           Term([make_tensor('H', 'v2,v3', 'g0,c3'),
                 make_tensor('t', 'c0,c3', 'v0,v1'), make_tensor('t', 'c1,c2', 'v2,v3')],
                make_sq('g0,v0,v1', 'c0,c1,c2'), 0.125),
           Term([make_tensor('H', 'c2,c3', 'g0,v3'),
                 make_tensor('t', 'c0,c1', 'v2,v3'), make_tensor('t', 'c2,c3', 'v0,v1')],
                make_sq('v0,v1,v2', 'g0,c0,c1'), -0.125),
           Term([make_tensor('H', 'c2,c3', 'g0,v3'),
                 make_tensor('t', 'c0,c2', 'v0,v1'), make_tensor('t', 'c1,c3', 'v2,v3')],
                make_sq('v0,v1,v2', 'g0,c0,c1'), 0.5)
           }

    for i in a:
        assert i in ref


def test_single_commutator():
    h = hamiltonian_operator(2)
    t2e = cluster_operator(2)

    a = single_commutator(h, t2e, for_commutator=True)
    b = single_commutator(h, t2e, for_commutator=False, n_process=4)
    assert a == b


def test_recursive_single_commutator_1():
    h = hamiltonian_operator(2)
    t2e = cluster_operator(2)

    a = recursive_single_commutator([h, t2e], 3, (0, 6), n_process=4)
    b = single_commutator(h, t2e, for_commutator=True)
    assert a[1] == b


def test_recursive_single_commutator_2():
    # Jacobi identity [[X, Y], Z] = [X, [Y, Z]] + [Y, [Z, X]] = [[X, Z], Y] + [[Z, Y], X]
    h = hamiltonian_operator(2)

    t1e = cluster_operator(1, start=4, hole_label='c', particle_label='v')
    t2e = cluster_operator(2, start=0, hole_label='c', particle_label='v')
    a = recursive_single_commutator([h, t2e, t1e], 1, (0, 6), n_process=4)

    t1e = cluster_operator(1, start=0, hole_label='c', particle_label='v')
    t2e = cluster_operator(2, start=4, hole_label='c', particle_label='v')
    b = recursive_single_commutator([h, t1e, t2e], 1, (0, 6), n_process=4)

    assert a[2] == b[2]  # note [t2e, t1e] = 0 for single reference


def test_recursive_single_commutator_3():
    # Jacobi identity [[X, Y], Z] = [X, [Y, Z]] + [Y, [Z, X]] = [[X, Z], Y] + [[Z, Y], X]
    h = hamiltonian_operator(2)

    t1e = cluster_operator(1, start=4)
    t2e = cluster_operator(2, start=0)
    a = recursive_single_commutator([h, t2e, t1e], 3, (0, 6), n_process=4)

    t1e = cluster_operator(1, start=0)
    t2e = cluster_operator(2, start=4)
    b = recursive_single_commutator([h, t1e, t2e], [2, 3], [(0, 4), (0, 6)], n_process=4)

    c = recursive_single_commutator([t1e, t2e, h], [2, 3], (0, 6), n_process=4)

    assert a[2] == combine_terms(b[2] + c[2])


def test_bch_cc_rsc_1():
    # single-reference CCSD with recursive single commutator approximation
    a = bch_cc_rsc(4, [1, 2], 1, (0, 4), single_reference=True, unitary=False)
    r = sorted(i for n, terms in a.items() for i in terms)
    assert len(r) == 46

    samples = [Term([make_tensor('H', 'v1,v2', 'g0,c2'), make_tensor('t', 'c2', 'v1'),
                     make_tensor('t', 'c0,c1', 'v0,v2')], make_sq('g0,v0', 'c0,c1'), 0.25),
               Term([make_tensor('H', 'c1,c2', 'g0,v2'), make_tensor('t', 'c1', 'v2'),
                     make_tensor('t', 'c0,c2', 'v0,v1')], make_sq('v0,v1', 'g0,c0'), -0.25),
               Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c0,c1', 'v0,v2'),
                     make_tensor('t', 'c2,c3', 'v1,v3')], make_sq('v0,v1', 'c0,c1'), -0.125),
               Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c0,c2', 'v0,v1'),
                     make_tensor('t', 'c1,c3', 'v2,v3')], make_sq('v0,v1', 'c0,c1'), -0.125),
               Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c0', 'v2'),
                     make_tensor('t', 'c2', 'v3'), make_tensor('t', 'c1,c3', 'v0,v1')],
                    make_sq('v0,v1', 'c0,c1'), -0.25),
               Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c2', 'v0'),
                     make_tensor('t', 'c3', 'v2'), make_tensor('t', 'c0,c1', 'v1,v3')],
                    make_sq('v0,v1', 'c0,c1'), -0.25)]
    for i in samples:
        assert i in r


def test_nested_cc_1():
    # single-reference CCSD equations
    a = [i for n in range(1, 5)
         for i in nested_commutator_cc(n, [1, 2], 1, max_n_open=4, single_reference=True)]

    # comparing to CCSD with recursive single commutator
    b = bch_cc_rsc(4, [1, 2], 1, (0, 4), single_reference=True, unitary=False)
    c = sorted(i for n, terms in b.items() for i in terms)
    assert len(a) == len(c)

    d = combine_terms(a + [Term.from_term(i, flip_sign=True) for i in c])
    ref = [Term([make_tensor('H', 'v1,v2', 'g0,c2'), make_tensor('t', 'c2', 'v1'),
                 make_tensor('t', 'c0,c1', 'v0,v2')], make_sq('g0,v0', 'c0,c1'), 0.25),
           Term([make_tensor('H', 'c1,c2', 'g0,v2'), make_tensor('t', 'c1', 'v2'),
                 make_tensor('t', 'c0,c2', 'v0,v1')], make_sq('v0,v1', 'g0,c0'), -0.25),
           Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c0,c1', 'v0,v2'),
                 make_tensor('t', 'c2,c3', 'v1,v3')], make_sq('v0,v1', 'c0,c1'), -0.125),
           Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c0,c2', 'v0,v1'),
                 make_tensor('t', 'c1,c3', 'v2,v3')], make_sq('v0,v1', 'c0,c1'), -0.125),
           Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c0', 'v2'),
                 make_tensor('t', 'c2', 'v3'), make_tensor('t', 'c1,c3', 'v0,v1')],
                make_sq('v0,v1', 'c0,c1'), -0.25),
           Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c2', 'v0'),
                 make_tensor('t', 'c3', 'v2'), make_tensor('t', 'c0,c1', 'v1,v3')],
                make_sq('v0,v1', 'c0,c1'), -0.25)]
    for i in d:
        assert i in ref
    assert len(d) == len(ref)


def test_nested_cc_2():
    # single-reference CCSDT equations
    a = [i for n in range(1, 5)
         for i in nested_commutator_cc(n, [1, 2, 3], 1, max_n_open=6, single_reference=True, n_process=4)]

    print_terms_ambit_functions(a)

    # # comparing to CCSD with recursive single commutator
    # b = bch_cc_rsc(4, [1, 2], 1, (0, 4), single_reference=True, unitary=False)
    # c = sorted(i for n, terms in b.items() for i in terms)
    # assert len(a) == len(c)
    #
    # d = combine_terms(a + [Term.from_term(i, flip_sign=True) for i in c])
    # ref = [Term([make_tensor('H', 'v1,v2', 'g0,c2'), make_tensor('t', 'c2', 'v1'),
    #              make_tensor('t', 'c0,c1', 'v0,v2')], make_sq('g0,v0', 'c0,c1'), 0.25),
    #        Term([make_tensor('H', 'c1,c2', 'g0,v2'), make_tensor('t', 'c1', 'v2'),
    #              make_tensor('t', 'c0,c2', 'v0,v1')], make_sq('v0,v1', 'g0,c0'), -0.25),
    #        Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c0,c1', 'v0,v2'),
    #              make_tensor('t', 'c2,c3', 'v1,v3')], make_sq('v0,v1', 'c0,c1'), -0.125),
    #        Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c0,c2', 'v0,v1'),
    #              make_tensor('t', 'c1,c3', 'v2,v3')], make_sq('v0,v1', 'c0,c1'), -0.125),
    #        Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c0', 'v2'),
    #              make_tensor('t', 'c2', 'v3'), make_tensor('t', 'c1,c3', 'v0,v1')],
    #             make_sq('v0,v1', 'c0,c1'), -0.25),
    #        Term([make_tensor('H', 'c2,c3', 'v2,v3'), make_tensor('t', 'c2', 'v0'),
    #              make_tensor('t', 'c3', 'v2'), make_tensor('t', 'c0,c1', 'v1,v3')],
    #             make_sq('v0,v1', 'c0,c1'), -0.25)]
    # for i in d:
    #     assert i in ref
    # assert len(d) == len(ref)


def test_print_ambit_functions():
    a = [i for n in range(1, 5)
         for i in nested_commutator_cc(n, [1, 2], 1, max_n_open=4, single_reference=True)]

    save_terms_ambit_functions(a, 'abc')


# def test_contraction_categorized_5():
#     from timeit import default_timer as timer
#
#     """
#     [a^{ v_{0} v_{1} v_{2} v_{3} }_{ c_{0} c_{1} c_{2} c_{3} }, a^{ v_{12} v_{13} }_{ c_{12} c_{13} }]
#     [a^{ v_{12} v_{13} }_{ c_{12} c_{13} }, a^{ v_{0} v_{1} v_{2} v_{3} }_{ c_{0} c_{1} c_{2} c_{3} }]
#     [a^{ v_{0} v_{1} v_{2} v_{3} }_{ c_{0} c_{1} c_{2} c_{3} }, a^{ v_{12} v_{13} }_{ c_{12} c_{13} }]
#     [a^{ v_{12} v_{13} }_{ c_{12} c_{13} }, a^{ v_{0} v_{1} v_{2} v_{3} }_{ c_{0} c_{1} c_{2} c_{3} }]
#     [a^{ v_{0} v_{1} v_{2} v_{3} }_{ c_{0} c_{1} c_{2} c_{3} }, a^{ v_{12} v_{13} }_{ c_{12} c_{13} }]
#     [a^{ v_{12} v_{13} }_{ c_{12} c_{13} }, a^{ v_{0} v_{1} v_{2} v_{3} }_{ c_{0} c_{1} c_{2} c_{3} }]
#     """
#
#     "a^{ g_{0} v_{0} v_{1} v_{2} v_{3} }_{ c_{0} c_{1} c_{2} c_{3} c_{4} }, a^{ v_{12} v_{13} }_{ c_{12} c_{13} }"
#     h = SQ("g0,v0,v1,v2,v3", "c0,c1,c2,c3,c4")
#     t = SQ("v12,v13", "c12,c13")
#
#     start = timer()
#     a = [i for con in compute_operator_contractions([h, t], max_cu=1, for_commutator=True, max_n_open=12,
#                                                     n_process=4, batch_size=0)
#          for i in con]
#     print(f"Time to compute T2^+ * H * T2 * T2: {timer() - start:.3f} s")
#     print(len(a))

# def test_f_t1():
#     F = hamiltonian_operator(1)
#     T1e = cluster_operator(1)
#     comm = commutator([F, T1e])
#     print_results(comm)
#     # print_results(comm, form='ambit')
#
#
# def test_f_t2():
#     F = hamiltonian_operator(1)
#     T2e = cluster_operator(2)
#     comm = commutator([F, T2e])
#     print_results(comm)
#     # print_results(comm, form='ambit')
#
#
# def test_v_t1():
#     V = hamiltonian_operator(2)
#     T1e = cluster_operator(1)
#     comm = commutator([V, T1e])
#     print_results(comm)
#     # print_results(comm, form='ambit')
#
#
# def test_v_t2():
#     V = hamiltonian_operator(2)
#     T2e = cluster_operator(2)
#     comm = commutator([V, T2e])
#     print_results(comm)
#     # print_results(comm, form='ambit')
#
#
# def test_v_a2():
#     V = hamiltonian_operator(2)
#     T2e = cluster_operator(2)
#     T2d = cluster_operator(2, de_excitation=True)
#     comm = commutator([V, T2e], min_n_open=6)
#     for i, terms in commutator([V, T2d], min_n_open=6).items():
#         comm[i] += terms
#
#     # results = defaultdict(list)
#     # for term in comm[6]:
#     #     comm_e = commutator([term, cluster_operator(2, start=2)], max_cu=1)
#     #     comm_d = commutator([term, cluster_operator(2, start=2, de_excitation=True)], max_cu=1)
#     #     for i, v in comm_e:
#     #         results[i] += v
#     #     for i, v in comm_d:
#     #         results[i] += v
#     #
#     # for i, v in results.items():
#     #     results[i] = combine_terms(v, presorted=False)
#     #
#     # print_results(results)
#
#
# def test_nested_comm_cc():
#     nested_commutator_cc(2, [1,2])
