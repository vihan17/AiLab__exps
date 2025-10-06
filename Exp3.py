import random
import itertools
from collections import defaultdict

def generate_k_sat(k, m, n):
    formula = []
    for _ in range(m):
        vars_ = random.sample(range(1, n+1), k)
        clause = [v if random.random() > 0.5 else -v for v in vars_]
        formula.append(clause)
    return formula

def is_satisfied(clause, assign):
    for lit in clause:
        if lit > 0:
            if assign[lit]:
                return True
        else:
            if not assign[-lit]:
                return True
    return False

def compute_satisfied(formula, assign):
    return sum(is_satisfied(c, assign) for c in formula)

def hill_climbing(formula, n, alpha, max_steps, eval_count):
    assign = [False] + [random.choice([True, False]) for _ in range(n)]
    curr_sat = compute_satisfied(formula, assign)
    eval_count[0] += 1
    steps = 0
    m_len = len(formula)
    if curr_sat == m_len:
        return True, steps, eval_count[0]
    while steps < max_steps:
        best_score = -float('inf')
        best_var = None
        best_delta = -float('inf')
        old_sats = [is_satisfied(c, assign) for c in formula]
        for i in range(1, n+1):
            assign[i] = not assign[i]
            new_sats = [is_satisfied(c, assign) for c in formula]
            eval_count[0] += 1
            make = sum(1 for j in range(m_len) if new_sats[j] and not old_sats[j])
            break_ = sum(1 for j in range(m_len) if old_sats[j] and not new_sats[j])
            delta = make - break_
            score = make - alpha * break_
            if score > best_score:
                best_score = score
                best_var = i
                best_delta = delta
            assign[i] = not assign[i]
        if best_delta <= 0:
            break
        assign[best_var] = not assign[best_var]
        curr_sat += best_delta
        steps += 1
        if curr_sat == m_len:
            return True, steps, eval_count[0]
    return curr_sat == m_len, steps, eval_count[0]

def beam_search(formula, n, w, alpha, max_steps, eval_count):
    m_len = len(formula)
    beams = []
    for _ in range(w):
        assign = [False] + [random.choice([True, False]) for _ in range(n)]
        sat = compute_satisfied(formula, assign)
        eval_count[0] += 1
        beams.append((sat, assign))
    best_sat = max(sat for sat, _ in beams)
    steps = 0
    if best_sat == m_len:
        return True, steps, eval_count[0]
    while steps < max_steps:
        candidates = []
        for curr_sat, assign in beams:
            old_sats = [is_satisfied(c, assign) for c in formula]
            for i in range(1, n+1):
                assign[i] = not assign[i]
                new_sats = [is_satisfied(c, assign) for c in formula]
                new_sat = sum(new_sats)
                eval_count[0] += 1
                make = sum(1 for j in range(m_len) if new_sats[j] and not old_sats[j])
                break_ = sum(1 for j in range(m_len) if old_sats[j] and not new_sats[j])
                delta = make - break_
                score = make - alpha * break_
                candidates.append((score, new_sat, list(assign), delta))
                assign[i] = not assign[i]
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        new_beams = []
        for _, new_sat, assign, _ in candidates[:w]:
            new_beams.append((new_sat, assign))
        beams = new_beams
        new_best = max(sat for sat, _ in beams)
        if new_best == m_len:
            return True, steps +1, eval_count[0]
        if new_best <= best_sat:
            break
        best_sat = new_best
        steps +=1
    return best_sat == m_len, steps, eval_count[0]

def vnd(formula, n, alpha, max_steps, eval_count):
    assign = [False] + [random.choice([True, False]) for _ in range(n)]
    curr_sat = compute_satisfied(formula, assign)
    eval_count[0] +=1
    steps = 0
    m_len = len(formula)
    if curr_sat == m_len:
        return True, steps, eval_count[0]
    while steps < max_steps:
        improved = False
        for k in range(1, 4):
            best_score = -float('inf')
            best_flips = None
            best_delta = -float('inf')
            old_sats = [is_satisfied(c, assign) for c in formula]
            for flips in itertools.combinations(range(1, n+1), k):
                for i in flips:
                    assign[i] = not assign[i]
                new_sats = [is_satisfied(c, assign) for c in formula]
                eval_count[0] +=1
                make = sum(1 for j in range(m_len) if new_sats[j] and not old_sats[j])
                break_ = sum(1 for j in range(m_len) if old_sats[j] and not new_sats[j])
                delta = make - break_
                score = make - alpha * break_
                if score > best_score:
                    best_score = score
                    best_flips = flips
                    best_delta = delta
                for i in flips:
                    assign[i] = not assign[i]
            if best_delta >0:
                for i in best_flips:
                    assign[i] = not assign[i]
                curr_sat += best_delta
                steps +=1
                improved = True
                break
        if not improved:
            break
        if curr_sat == m_len:
            return True, steps, eval_count[0]
    return curr_sat == m_len, steps, eval_count[0]

def run_algo(formula, n, algo_type, w, alpha, max_steps, eval_count):
    if algo_type == 'hill':
        return hill_climbing(formula, n, alpha, max_steps, eval_count)
    elif algo_type == 'beam':
        return beam_search(formula, n, w, alpha, max_steps, eval_count)
    elif algo_type == 'vnd':
        return vnd(formula, n, alpha, max_steps, eval_count)
import numpy as np  # Colab has numpy pre-installed

def run_experiments(n, m_values, num_instances=3, num_runs=3, max_steps=100):
    results = {}
    for m in m_values:
        success_rates = {'hill1': [], 'hill2': [], 'beam3_1': [], 'beam3_2': [], 'beam4_1': [], 'beam4_2': [], 'vnd1': [], 'vnd2': []}
        for _ in range(num_instances):
            formula = generate_k_sat(3, m, n)
            for run in range(num_runs):
                for algo in ['hill', 'beam3', 'beam4', 'vnd']:
                    alpha_values = [1, 2]
                    for alpha in alpha_values:
                        eval_count = [0]
                        if algo == 'hill':
                            success, steps, evals = hill_climbing(formula, n, alpha, max_steps, eval_count)
                            key = f'hill{alpha}'
                        elif algo.startswith('beam'):
                            w = 3 if '3' in algo else 4
                            success, steps, evals = beam_search(formula, n, w, alpha, max_steps, eval_count)
                            key = f'beam{w}_{alpha}'
                        else:
                            success, steps, evals = vnd(formula, n, alpha, max_steps, eval_count)
                            key = f'vnd{alpha}'
                        success_rates[key].append(success)
        for key in success_rates:
            results[(m, key)] = np.mean(success_rates[key])
    print(results)  # Or format into a table

run_experiments(n=20, m_values=[40, 80, 120])
