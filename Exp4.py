import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

def load_octave_matrix(file_path):
    print(f"Reading Octave-format file from: {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header_end = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith('# ndims:'):
            header_end = idx + 2
            break
    if header_end == 0:
        raise ValueError("Missing '# ndims:' information in the file.")

    pixels = [line.strip() for line in lines[header_end:]]
    pixel_data = np.fromstring(" ".join(pixels), dtype=np.uint8, sep=' ')
    img_size = 512

    if pixel_data.size != img_size * img_size:
        raise ValueError(f"Invalid size. Expected {img_size*img_size}, got {pixel_data.size}.")
    return pixel_data.reshape((img_size, img_size))

class PuzzleSolver:
    def __init__(self, input_image, grid_size=4):
        self.input_image = input_image
        self.grid_size = grid_size
        self.pieces, self.piece_shape = self._divide_blocks()
        self.total_pieces = len(self.pieces)
        self.order = list(range(self.total_pieces))

    def _divide_blocks(self):
        height, width = self.input_image.shape
        block_h, block_w = height // self.grid_size, width // self.grid_size
        blocks = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                blocks.append(self.input_image[x*block_h:(x+1)*block_h, y*block_w:(y+1)*block_w])
        return blocks, (block_h, block_w)

    def _compute_cost(self, arrangement):
        score = 0.0
        grid_layout = [[self.pieces[arrangement[i*self.grid_size + j]] for j in range(self.grid_size)] for i in range(self.grid_size)]
        
        # Horizontal mismatch
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                left, right = grid_layout[i][j], grid_layout[i][j+1]
                score += np.sum(np.abs(left[:, -1].astype(np.int32) - right[:, 0].astype(np.int32)))
        
        # Vertical mismatch
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                top, bottom = grid_layout[i][j], grid_layout[i+1][j]
                score += np.sum(np.abs(top[-1, :].astype(np.int32) - bottom[0, :].astype(np.int32)))
                
        return float(score)

    def solve(self, start_temp, decay_rate, swaps_per_temp, max_iters, rand_seed, show_step):
        if rand_seed is not None:
            random.seed(rand_seed)
        
        best_order = self.order[:]
        current_cost = self._compute_cost(self.order)
        lowest_cost = current_cost
        
        print(f"Initial mismatch score: {current_cost:.0f}")

        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        temperature = start_temp
        iteration = 0

        while iteration < max_iters and temperature > 0.1:
            for _ in range(swaps_per_temp):
                if iteration >= max_iters: break
                
                a, b = random.sample(range(self.total_pieces), 2)
                self.order[a], self.order[b] = self.order[b], self.order[a]
                
                new_cost = self._compute_cost(self.order)
                delta = new_cost - current_cost
                
                if delta < 0 or (temperature > 1e-9 and random.random() < math.exp(-delta / temperature)):
                    current_cost = new_cost
                    if current_cost < lowest_cost:
                        lowest_cost = current_cost
                        best_order = self.order[:]
                else:
                    self.order[a], self.order[b] = self.order[b], self.order[a]
                
                iteration += 1

                if iteration % show_step == 0:
                    self._draw(ax, fig, best_order, iteration, temperature, lowest_cost)
            
            temperature *= decay_rate

        plt.ioff()
        print(f"\nCompleted. Best score achieved: {lowest_cost:.0f}")
        return best_order, lowest_cost

    def _draw(self, ax, fig, arrangement, step, temperature, score):
        img = self.combine_image(arrangement)
        ax.clear()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Iter {step} | Temp {temperature:.2f}\nLowest score {score:.0f}")
        ax.axis('off')
        fig.canvas.draw()
        plt.pause(0.001)

    def combine_image(self, arrangement):
        block_h, block_w = self.piece_shape
        output = np.zeros((self.grid_size * block_h, self.grid_size * block_w), dtype=self.pieces[0].dtype)
        index = 0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                output[x*block_h:(x+1)*block_h, y*block_w:(y+1)*block_w] = self.pieces[arrangement[index]]
                index += 1
        return output

if __name__ == "__main__":
    try:
        scrambled = load_octave_matrix('scrambled_lena.mat').T

        if scrambled is not None:
            algo = PuzzleSolver(scrambled, grid_size=4)

            solution, score = algo.solve(
                start_temp=5500.0,
                decay_rate=0.999,
                swaps_per_temp=250,
                max_iters=50000,
                rand_seed=99,
                show_step=1000
            )

            output_image = algo.combine_image(solution)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(scrambled, cmap='gray')
            plt.title('Input (Scrambled)')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(output_image, cmap='gray')
            plt.title(f'Solved Puzzle (Energy {score:.0f})')
            plt.axis('off')
            plt.show()

    except Exception as err:
        print(f"Program failure: {err}")
