### Code to generate DSRG equations based on generalized normal ordering

#### Requirement
- Python 3.6
- SymPy
- pytest

#### Design Documentation

*Project Variables, Functions, Classes*
- `mo_space.py`: MO space info
- `integer_partition.py`: function to partition an integer


*Core Classes* 
- `Index.py`: Index class 
- `Indices.py`: Indices class
- `IndicesPair.py`: IndicesPair class
- `SQOperator.py`: SecondQuantizedOperator class 
- `Tensor.py`: Tensor class 
- `Term.py`: Term class

*Operator Contraction Functions*
- `sqop_contraction.py`: contractions of string of creation and annihilation operators
- `phys_contraction.py`: contractions of physical operators

#### Testing
Run command `python -m pytest tests`
