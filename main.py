from dataclasses import dataclass
from typing import List, Tuple, Optional
import pyomo.environ as pyo


@dataclass
class Rectangle:
    """
    Represents a rectangle with position and dimensions.
    """
    x: int  # top-left x coordinate
    y: int  # top-left y coordinate
    width: int
    height: int
    is_assigned: bool = False
    
    def __str__(self):
        return f"Rectangle({self.width}x{self.height} at ({self.x},{self.y}))"
    
    def overlaps_with(self, other: 'Rectangle') -> bool:
        """Check if this rectangle overlaps with another rectangle."""
        pass  # TODO: Implement overlap detection
    
    def covers_cell(self, x: int, y: int) -> bool:
        """Check if this rectangle covers a given cell."""
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


class GridCoveringPuzzle:
    """
    A puzzle where we need to cover an NxN grid with rectangles such that:
    1. No rectangles overlap
    2. Each row has exactly one empty cell
    3. Each column has exactly one empty cell
    """
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.rectangles: List[Rectangle] = []
        self.candidates: List[Rectangle] = self.create_candidates()
        self.cells: List[Tuple[int, int]] = self.create_cells()

    def create_candidates(self) -> List[Rectangle]:
        """Create all possible rectangle candidates for the puzzle."""
        candidates: List[Rectangle] = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for width in range(1, self.grid_size - x + 1):
                    for height in range(1, self.grid_size - y + 1):
                        candidates.append(Rectangle(x, y, width, height))
        return candidates

    def create_cells(self) -> List[Tuple[int, int]]:
        """Create all cells in the grid."""
        return [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
    
    def visualize_solution(self, rectangles):
        """Visualize the solution with rectangles."""
        print(f"\nGrid {self.grid_size}x{self.grid_size} Solution:")
        print("=" * (self.grid_size * 2 + 1))
        
        # Create grid representation
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark rectangles with different characters
        for idx, rect in enumerate(rectangles):
            char = chr(ord('A') + idx)  # Use letters A, B, C, etc.
            for x in range(rect.x, rect.x + rect.width):
                for y in range(rect.y, rect.y + rect.height):
                    grid[x][y] = char
        
        # Print the grid
        for row in grid:
            print("|" + "|".join(row) + "|")
        print("=" * (self.grid_size * 2 + 1))
    
    def solve(self, solver_name: str = 'appsi_highs'):
        """Solve the puzzle using optimization."""

        # create the pyomo model
        model = pyo.ConcreteModel()
        model.candidates = pyo.Set(initialize=range(len(self.candidates)))
        model.cells = pyo.Set(initialize=self.cells)
        model.rows = pyo.Set(initialize=range(self.grid_size))  
        model.columns = pyo.Set(initialize=range(self.grid_size))

        # V1. Assign a candidate
        model.assign = pyo.Var(model.candidates, domain=pyo.Binary)

        # C1. Each cell is covered by maximum one candidate
        def cover_rule(m, x, y):
            return sum(m.assign[candidate_idx] for candidate_idx in m.candidates if self.candidates[candidate_idx].covers_cell(x, y)) <= 1
        model.cover = pyo.Constraint(range(self.grid_size), range(self.grid_size), rule=cover_rule)

        #C2. Only one free cell per row
        def free_cell_per_row_rule(m, row):
            return sum(1 - sum(m.assign[candidate_idx] for candidate_idx in m.candidates if self.candidates[candidate_idx].covers_cell(row, y)) for y in range(self.grid_size)) == 1
        model.free_cell_per_row = pyo.Constraint(model.rows, rule=free_cell_per_row_rule)

        #C3. Only one free cell per column
        def free_cell_per_column_rule(m, col):
            return sum(1 - sum(m.assign[candidate_idx] for candidate_idx in m.candidates if self.candidates[candidate_idx].covers_cell(x, col)) for x in range(self.grid_size)) == 1
        model.free_cell_per_column = pyo.Constraint(model.columns, rule=free_cell_per_column_rule)

        # O1. Minimize nr of rectangles assigned 
        def objective_rule(m):
            return sum(m.assign[candidate_idx] for candidate_idx in m.candidates)
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Solve the model
        solver = pyo.SolverFactory(solver_name)
        results = solver.solve(model, tee=True)
        
        # Print the results
        print(f"\nSolver status: {results.solver.termination_condition}")
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("Optimal solution found!")
            
            # Extract selected rectangles
            selected_rectangles = []
            for candidate_idx in model.candidates:
                if model.assign[candidate_idx].value > 0.5:  # Binary variable > 0.5 means True
                    selected_rectangles.append(self.candidates[candidate_idx])
            
            print(f"\nSelected {len(selected_rectangles)} rectangles:")
            for i, rect in enumerate(selected_rectangles):
                print(f"  {i+1}. {rect}")
            
            # Visualize the solution
            self.visualize_solution(selected_rectangles)
        else:
            print("No optimal solution found.")

    def cover_rule(self, model, cell: Tuple[int, int]):
        """Constraint: Each cell is covered by maximum one candidate."""
        return sum(model.assign[candidate_idx] for candidate_idx in model.candidates if self.candidates[candidate_idx].covers_cell(cell[0], cell[1])) <= 1

    def free_cell_per_row_rule(self, model, row: int):
        """Constraint: Each row has exactly one free cell."""
        return sum(1 - sum(model.assign[candidate_idx] for candidate_idx in model.candidates if self.candidates[candidate_idx].covers_cell(row, y)) for y in range(self.grid_size)) == 1

    def free_cell_per_column_rule(self, model, column: int):
        """Constraint: Each column has exactly one free cell."""
        return sum(1 - sum(model.assign[candidate_idx] for candidate_idx in model.candidates if self.candidates[candidate_idx].covers_cell(x, column)) for x in range(self.grid_size)) == 1

    def objective_rule(self, model):
        """Objective: Minimize the number of assigned candidates."""
        return sum(model.assign[candidate_idx] for candidate_idx in model.candidates)



def main():
    """Main function for testing."""
    puzzle = GridCoveringPuzzle(9)
    print(f"Created puzzle with grid size: {puzzle.grid_size}")
    puzzle.solve(solver_name='cbc')


if __name__ == "__main__":
    main()