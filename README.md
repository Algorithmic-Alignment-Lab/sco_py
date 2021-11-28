[![continuous-integration](https://github.com/Algorithmic-Alignment-Lab/sco_py/actions/workflows/ci.yaml/badge.svg)](https://github.com/Algorithmic-Alignment-Lab/sco_py/actions/workflows/ci.yaml)

# sco_py: Sequential Convex Optimization with Python
sco_py is a lightweight Sequential Convex Optimization library for solving non-convex optimization problems. sco_py is intended for use with the OpenTAMP planning system. Currently, the library supports both [Gurobi](https://www.gurobi.com/) (license required) and [OSQP](https://osqp.org/) (open-source, no license required!) as backend QP solvers.

## Installation
### From PyPI with pip
Simply run: `pip install sco-py`

### From GitHub with pip
Simply run: `pip install git+https://github.com/Algorithmic-Alignment-Lab/sco_py.git`

### Developer (from source)
1. Clone this repository [from GitHub](https://github.com/Algorithmic-Alignment-Lab/sco_py)
1. Install Poetry by following the instructions from [here](https://python-poetry.org/docs/#installation)
1. Install all dependencies with `poetry install`.

## Contributing
sco is an open-source repository and as such, we welcome contributions from interested members of the community! If you have a new idea for a feature/contribution, do post in the ['Discussions' tab of the GitHub repository](https://github.com/Algorithmic-Alignment-Lab/sco_py/discussions) to get some feedback from the maintainers before starting development. In general, we recommend that you fork the main repository, create a new branch with your proposed change, then open a pull-request into the main repository. The main requirement for a new feature is that it cannot break the current test cases (see below for how to run our tests) unless this is unavoidable, and in this case, it should modify/introduce new tests as necessary. In particular, we welcome updates to documentation (docstrings, comments, etc. that make the codem more approachable for new users) and test cases!

### Running tests
If you do not have a license for Gurobi, then you can only run the OSQP tests. To do so, run:
```
pytest tests/sco_osqp/
```
If you do have a license for Gurobi, then you can run all tests with the `pytest` command.

Note that our Contrinuous Integration (CI) setup only checks and reports status for OSQP tests. In general, if you are contributing a new feature, it *must* pass the existing OSQP tests and contribute new tests that test the new feature at least with OSQP (and preferably with Gurobi as well).