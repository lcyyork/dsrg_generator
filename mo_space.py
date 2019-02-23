# spin-orbital space
so_space = ['g', 'p', 'h', 'v', 'c', 'a']
space_priority_so = {k: v for v, k in enumerate(so_space)}
space_relation_so = {'c': {'c'}, 'a': {'a'}, 'v': {'v'},
                     'p': {'a', 'v'}, 'h': {'c', 'a'},
                     'g': {'c', 'a', 'v'}
                     }

# spin-integrated alpha subspace
space_priority_a = space_priority_so
space_relation_a = space_relation_so

# spin-integrated beta subspace
mo_space_b = [i.upper() for i in so_space]
space_priority_b = {k: v for v, k in enumerate(mo_space_b)}
space_relation_b = {'C': {'C'}, 'A': {'A'}, 'V': {'V'},
                    'P': {'A', 'V'}, 'H': {'C', 'A'},
                    'G': {'C', 'A', 'V'}
                    }

# spin-integrated space
mo_space = so_space + mo_space_b
space_priority = {k: v for v, k in enumerate(mo_space)}
space_relation = {**space_relation_a, **space_relation_b}
