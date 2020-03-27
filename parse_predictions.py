import os
from collections import defaultdict
from statistics import mean, stdev


folder = "predictions_output"

results = defaultdict(lambda: defaultdict(list))

for fp in os.listdir(folder):
    with open(os.path.join(folder, fp), 'r') as f:
        model = f.readline().strip()
        task = f.readline().strip()
        f.readline()
        resultlines = list()
        for line in f.readlines():
            if '%' in line:
                continue
            else:
                resultlines.append(line.strip())
        for i in range(len(resultlines) // 3):
            i *= 3
            lines = resultlines[i:i+3]
            formline = ''.join([ch for ch in lines[0] if ch.isalpha() or ch == ' ']).split()
            form = '_'.join([s for i, s in enumerate(formline) if i % 2 == 1])
            score = float(lines[1])
            results[task][form].append(score)

for task, forms in results.items():
    for form in forms:
        results[task][form] = (mean(results[task][form]), stdev(results[task][form]), max(results[task][form]), min(results[task][form]))
        print(f'{task} {form} mean: {results[task][form][0]:.2f}, std: {results[task][form][1]:.2f}, max: {results[task][form][2]:.2f}, min: {results[task][form][3]:.2f}')
        # results[task][form] = (mean(results[task][form]), stdev(results[task][form]))
        # print(f'{task} {form} mean: {results[task][form][0]:.3f}, std: {results[task][form][1]:.3f}')
        # results[task][form] = mean(results[task][form])
        # print(task, form, results[task][form])


