"""Run the paper analysis.

    python analysis/run_analysis.py JOB [JOB ...]     (JOB = all, or one of the names in JOBS)

The analysis was developed in a notebook. Each job executes a list of code files
(cells/ = notebook cells, steps/ = analysis steps) in order, in one shared
namespace, as the notebook did. Paths come from analysis/config.py. The output of
every step is logged to OUT_DIR/logs/; a job stops at the first error.
"""
import contextlib
import io
import os
import sys
import time
import traceback
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config  # noqa: E402

# load data and the evaluation mask (TRENDY v13 S2 and the experiment-1 emulators, 2010-2023)
BASE = ['cells/c00_imports.py', 'cells/c01_models.py', 'cells/c02_mask_helper.py', 'cells/c03_definitions.py',
        'cells/c05_load.py', 'cells/c06_evaluation_mask.py']

JOBS = {
    # Figs 2-6, Supp Figs 1 and 3-9, Fig3_R2_two_scales.txt, Fig5_spatial_R2.txt, Fig6_tail_underestimation.txt
    'figures': BASE + ['cells/c14_formatter.py', 'cells/c16_land_mask.py',
                       'cells/c19_fig2.py', 'cells/c20_suppfig1.py',
                       'cells/c21a_fig4_functions.py', 'cells/c21b_fig4.py', 'cells/c22_suppfig3_5.py',
                       'cells/c23_suppfig6.py',
                       'cells/c24a_fig3_metrics.py', 'cells/c24b_fig3_compute.py', 'cells/c24c_fig3_plot.py',
                       'cells/c25_fig6.py', 'cells/c26_suppfig7_9.py', 'cells/c30_fig5.py'],
    # Supp Tables 4 and 5
    'tables': BASE + ['cells/c16_land_mask.py', 'cells/c107a_trendy_v14.py', 'cells/c108a_era5_baseline.py',
                      'cells/c108b_supptables4_5.py'],
    # Supp Table 6 (2024, experiment 2) and its subset-threshold sensitivity
    'table6': BASE + ['cells/c16_land_mask.py', 'cells/c107a_trendy_v14.py', 'cells/c108a_era5_baseline.py',
                      'cells/c109_supptable6.py', 'steps/supp_table6_subset_sensitivity.py'],
    # fit_for_purpose/P1-P3 and SuppFig2_values.csv
    'regions': BASE + ['steps/regional_fit_for_purpose.py'],
    # fit_for_purpose/P4_*
    'climatology': BASE + ['steps/climatology_baseline.py'],
    # model_differences/ (needs figures and tables)
    'model_differences': BASE + ['cells/c24a_fig3_metrics.py', 'cells/c24b_fig3_compute.py',
                                 'cells/c21a_fig4_functions.py', 'steps/model_differences.py'],
    # checks/: one-step vs recursive (C4) and the grid seasonal-cycle climatology baseline
    'onestep': BASE + ['cells/c16_land_mask.py', 'cells/c107a_trendy_v14.py', 'cells/c108a_era5_baseline.py',
                       'cells/c24a_fig3_metrics.py', 'steps/onestep_vs_recursive.py'],
    # key_numbers.txt
    'key_numbers': BASE + ['cells/c16_land_mask.py', 'cells/c34_africa_helpers.py', 'cells/c108a_era5_baseline.py',
                           'steps/key_numbers.py'],
    # Supp Table 3 (reads the outputs of the jobs above)
    'supp_table3': ['steps/supp_table3.py'],
    # Supp Fig 2 (reads SuppFig2_values.csv)
    'supp_fig2': BASE[:4] + ['steps/supp_fig2.py'],
    # source_data/: CSV source data of the figures and tables (needs the jobs above)
    'source_data': BASE + ['steps/source_data.py'],
}
# outputs of other jobs that a job reads
REQUIRES = {
    'model_differences': ['Fig6_tail_underestimation.txt', 'SuppTable5.csv'],
    'supp_table3': ['SuppTable5.csv', 'fit_for_purpose/P3_regional_summary.csv', 'fit_for_purpose/P4_3_annual_error_long.csv',
                    'model_differences/Part1_Fig3_column_summary.csv', 'checks/grid_seasonal_cycle_climatology_R2.csv',
                    'key_numbers.txt'],
    'supp_fig2': ['SuppFig2_values.csv'],
    'source_data': ['model_differences/Part1_Fig3_heatmap_values.csv', 'model_differences/Part1_Fig2_SuppFig1_global.csv',
                    'model_differences/Part1_Fig4_SuppFig3-5_panel_stats.csv', 'Fig5_spatial_R2.txt', 'SuppFig2_values.csv',
                    'SuppTable3.csv', 'SuppTable4.csv', 'SuppTable5.csv', 'SuppTable6.csv'],
}
ORDER = ['figures', 'tables', 'table6', 'regions', 'climatology', 'model_differences', 'onestep', 'key_numbers',
         'supp_table3', 'supp_fig2', 'source_data']

def run(job):
    missing = [f for f in REQUIRES.get(job, []) if not os.path.exists(config.OUT_DIR + f)]
    if missing:
        sys.exit(f'{job}: run the jobs that write {missing} first (or use "all")')
    t0 = time.time()
    log_dir = config.OUT_DIR + 'logs/'
    os.makedirs(log_dir, exist_ok=True)
    ns = {'__name__': '__main__', '__builtins__': __builtins__}
    ns.update({name: getattr(config, name) for name in config.NAMES})
    for rel in JOBS[job]:
        code = open(os.path.join(HERE, rel), encoding='utf-8').read()
        out, err = io.StringIO(), io.StringIO()
        exc = None
        t1 = time.time()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                exec(compile(code, rel, 'exec'), ns)
            except Exception:
                exc = traceback.format_exc()
        plt.close('all')
        name = rel.replace('/', '_').replace('.py', '')
        with open(log_dir + f'{job}__{name}.txt', 'w') as fh:
            fh.write(out.getvalue())
            fh.write(exc or '')
        status = 'ERROR: ' + exc.strip().splitlines()[-1] if exc else 'ok'
        print(f'[{time.time() - t0:6.0f}s] {job}: {rel} ({time.time() - t1:.0f}s) {status}', flush=True)
        if exc:
            print(out.getvalue()[-3000:], file=sys.stderr)
            print(exc, file=sys.stderr)
            sys.exit(1)
    print(f'[{time.time() - t0:6.0f}s] {job} done', flush=True)


if __name__ == '__main__':
    jobs = sys.argv[1:]
    if not jobs or any(j not in JOBS and j != 'all' for j in jobs):
        sys.exit('usage: run_analysis.py all | ' + ' | '.join(ORDER))
    for job in (ORDER if jobs == ['all'] else jobs):
        run(job)
