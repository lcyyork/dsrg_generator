# spin-orbital space
so_space = ['g', 'p', 'h', 'v', 'c', 'a']
space_priority_so = {k: v for k, v in zip(so_space, range(6))}
space_relation_so = {'c': {'c'}, 'a': {'a'}, 'v': {'v'},
                     'p': {'a', 'v'}, 'h': {'c', 'a'},
                     'g': {'c', 'a', 'v'}
                     }

# spin-integrated space
mo_space = so_space + [i.upper() for i in so_space]
space_priority = {k: v for k, v in zip(mo_space, range(12))}
space_relation = {'c': {'c'}, 'a': {'a'}, 'v': {'v'}, 'C': {'C'}, 'A': {'A'}, 'V': {'V'},
                  'p': {'a', 'v'}, 'h': {'c', 'a'}, 'P': {'A', 'V'}, 'H': {'C', 'A'},
                  'g': {'c', 'a', 'v'}, 'G': {'C', 'A', 'V'}
                  }
