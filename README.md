### Code to generate DSRG equations based on generalized normal ordering

#### Requirement
- Python 3.6
- SymPy
- pytest

#### Design Documentation
- MO space info: mo_space.py
- Index class: Index.py
- Indices class: Indices.py
- IndicesPair class: IndicesPair.py
- SecondQuantizedOperator class: SQOperator.py
- Tensor class: Tensor.py
- Term class: Term.py

- sqop_contraction.py: contractions of string of creation and annihilation operators
- phys_contraction.py: contractions of physical operators

#### Testing
Run command `python -m pytest tests`
