from argparse import ArgumentParser
import pandas as pd
from pathlib import Path


def main():
    parser = ArgumentParser("Script to aggregate MIA results")
    parser.add_argument('-d', '--results_dir', help='/path/to/results/directory')
    parser.add_argument('-dt', '--distinguisher_type', help='Type of distinguisher to get results for')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/directory')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    results_dfs = []
    for unlearn_problem_dir in results_dir.iterdir():
        if unlearn_problem_dir.is_dir():
            unlearn_problem = unlearn_problem_dir.name
            results_csv = list(unlearn_problem_dir.glob(f'{args.distinguisher_type}_results.csv'))[0]
            results_df = pd.read_csv(str(results_csv))
            results_df.insert(0, 'unlearn_problem', unlearn_problem)
            results_dfs.append(results_df)
    
    df = pd.concat(results_dfs)
    df.to_csv(str(output_dir /f'{args.distinguisher_type}_results.csv'), index=False)


if __name__ == "__main__":
    main()
