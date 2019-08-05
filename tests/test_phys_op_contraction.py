from collections import defaultdict
from timeit import default_timer as timer
from src.phys_op_contraction import contract_terms, single_commutator, print_results
from src.Term import Term, hamiltonian_operator, cluster_operator
from src.Tensor import Tensor
from src.SQOperator import SecondQuantizedOperator
# from src.phys_op_contraction import nested_commutator_ucc


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
