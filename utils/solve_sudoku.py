''' Solves sudoku using backtracking '''
class Solve_Sudoku:
    def __init__(self, sudoku_size=9):
        assert((sudoku_size**0.5) - int(sudoku_size**0.5) == 0)
        self.N = sudoku_size
        self.box_size = int(self.N**0.5)

    def print_sudoku(self, sudoku): 
        print("\nSolved!!!")
        for i in range(self.N): 
            for j in range(self.N): 
                print(sudoku[i][j], end=" ")
            print ('')
 
    def find_empty_location(self, sudoku, pos): 
        for row in range(self.N): 
            for col in range(self.N): 
                if(sudoku[row][col] == 0): 
                    pos[0], pos[1] = row, col 
                    return True
        return False

    def used_in_row(self, sudoku, row, digit): 
        for i in range(self.N): 
            if(sudoku[row][i] == digit): 
                return True
        return False

    def used_in_col(self, sudoku, col, digit): 
        for i in range(self.N): 
            if(sudoku[i][col] == digit): 
                return True
        return False

    def used_in_box(self, sudoku, row, col, digit): 
        for i in range(self.box_size): 
            for j in range(self.box_size): 
                if(sudoku[i + row][j + col] == digit): 
                    return True
        return False

 
    def check_location_is_safe(self, sudoku, row, col, digit): 
        # Check if 'digit' is not already placed in current row, 
        # current column and current 3x3 box 
        return not self.used_in_row(sudoku, row, digit) and not self.used_in_col(sudoku, col, digit) and not self.used_in_box(sudoku, row - row % 3, col - col % 3, digit) 

    def solve_sudoku(self, sudoku): 
        # 'pos' keeps the record of location of empty cell
        pos =[0, 0] 
        
        # If there is no unassigned location, we are done	 
        if(not self.find_empty_location(sudoku, pos)): 
            return sudoku
        
        # Assigning list values to row and col that we got from the above Function 
        row = pos[0] 
        col = pos[1] 

        # consider digits 1 to self.N 
        for digit in range(1, self.N + 1): 
            
            # if looks promising 
            if(self.check_location_is_safe(sudoku, row, col, digit)): 

                # make tentative assignment 
                sudoku[row][col]= digit 

                # return, if success, ya ! 
                if(self.solve_sudoku(sudoku)): 
                    return sudoku

                # failure, unmake & try again 
                sudoku[row][col] = 0
                
        # this triggers backtracking		 
        return False