import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# List available solvers
print('Available solvers:')
for solver_name in ['cbc', 'glpk', 'gurobi', 'cplex', 'highs', 'appsi_highs']:
    try:
        solver = SolverFactory(solver_name)
        if solver.available():
            print(f'✓ {solver_name}: Available')
        else:
            print(f'✗ {solver_name}: Not available')
    except:
        print(f'✗ {solver_name}: Error checking')