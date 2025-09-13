import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def create_simple_mip():
    """
    Create a simple Mixed Integer Programming model.
    This is a classic production planning problem with binary variables.
    """
    # Create the model
    model = pyo.ConcreteModel()
    
    # Sets
    model.products = pyo.Set(initialize=['A', 'B', 'C'])
    model.machines = pyo.Set(initialize=['M1', 'M2', 'M3'])
    
    # Parameters
    # Profit per unit of each product
    model.profit = pyo.Param(model.products, initialize={'A': 10, 'B': 15, 'C': 12})
    
    # Machine capacity (hours available)
    model.capacity = pyo.Param(model.machines, initialize={'M1': 40, 'M2': 35, 'M3': 30})
    
    # Time required to produce one unit of each product on each machine
    model.time_required = pyo.Param(
        model.products, model.machines,
        initialize={
            ('A', 'M1'): 2, ('A', 'M2'): 3, ('A', 'M3'): 1,
            ('B', 'M1'): 1, ('B', 'M2'): 2, ('B', 'M3'): 2,
            ('C', 'M1'): 3, ('C', 'M2'): 1, ('C', 'M3'): 3,
        }
    )
    
    # Variables
    # Binary variable: 1 if product i is produced, 0 otherwise
    model.produce = pyo.Var(model.products, domain=pyo.Binary)
    
    # Continuous variable: quantity of each product to produce
    model.quantity = pyo.Var(model.products, domain=pyo.NonNegativeReals)
    
    # Objective: maximize total profit
    model.objective = pyo.Objective(
        expr=sum(model.profit[i] * model.quantity[i] for i in model.products),
        sense=pyo.maximize
    )
    
    # Constraints
    # Machine capacity constraints
    def machine_capacity_rule(model, m):
        return sum(
            model.time_required[i, m] * model.quantity[i] 
            for i in model.products
        ) <= model.capacity[m]
    
    model.machine_capacity = pyo.Constraint(model.machines, rule=machine_capacity_rule)
    
    # Logical constraint: can only produce if we decide to produce
    def production_logic_rule(model, i):
        return model.quantity[i] <= 1000 * model.produce[i]  # Big M constraint
    
    model.production_logic = pyo.Constraint(model.products, rule=production_logic_rule)
    
    # At least one product must be produced
    def at_least_one_rule(model):
        return sum(model.produce[i] for i in model.products) >= 1
    
    model.at_least_one = pyo.Constraint(rule=at_least_one_rule)
    
    return model


def solve_mip(model):
    """Solve the MIP model and display results."""
    # Create solver
    solver = SolverFactory('appsi_highs')  # HiGHS solver via appsi
    
    # Solve the model
    results = solver.solve(model, tee=True)
    
    # Check if solution is optimal
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("\n" + "="*50)
        print("OPTIMAL SOLUTION FOUND")
        print("="*50)
        
        print(f"\nTotal Profit: ${model.objective():.2f}")
        
        print("\nProduction Decision:")
        for i in model.products:
            if model.produce[i].value > 0.5:  # Binary variable > 0.5 means True
                print(f"  Product {i}: YES (Quantity: {model.quantity[i].value:.1f})")
            else:
                print(f"  Product {i}: NO")
        
        print("\nMachine Utilization:")
        for m in model.machines:
            used_time = sum(
                model.time_required[i, m] * model.quantity[i].value 
                for i in model.products
            )
            utilization = (used_time / model.capacity[m]) * 100
            print(f"  {m}: {used_time:.1f}/{model.capacity[m]} hours ({utilization:.1f}%)")
            
    else:
        print("No optimal solution found.")
        print(f"Solver status: {results.solver.termination_condition}")


def main():
    print("Creating a simple Mixed Integer Programming model...")
    
    # Create the MIP model
    model = create_simple_mip()
    
    print("Model created successfully!")
    print(f"Number of variables: {len(model.produce) + len(model.quantity)}")
    print(f"Number of constraints: {len(model.machine_capacity) + len(model.production_logic) + 1}")
    
    # Solve the model
    print("\nSolving the model...")
    solve_mip(model)


if __name__ == "__main__":
    main()

