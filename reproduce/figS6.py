"""
Fig S6: per-site model accuracy for embeddings from different MSA Transformer layers.
Reads fig3a_site_spearmans.tsv, written by preprint.py, for each layer's model.
    python reproduce/figS6.py --layer 1 figures/layer1 --layer 6 figures/layer6 --layer 12 figures/preprint \
        --out figures/figS6.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--layer', nargs=2, action='append', required=True, metavar=('LAYER', 'FOLDER'),
                help='a layer and the preprint.py output folder of its model (repeat for each layer)')
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()

fig, ax = plt.subplots(figsize=(9, 3))
for layer, folder in args.layer:
    rs = pd.read_csv(Path(folder) / 'fig3a_site_spearmans.tsv', sep='\t', index_col='site')['spearman']
    rs = rs.drop(index=0)                         # site 0 is the CLS token
    ax.plot(rs.index, rs.rolling(5, center=True, min_periods=1).mean(),
            label=f'layer {layer} (median {rs.median():.2f})')
ax.set(xlabel='Alignment position', ylabel='Spearman R (rolling mean)',
       title='Per-site model accuracy on the test set, by MSA Transformer layer')
ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
fig.savefig(args.out, dpi=300, bbox_inches='tight')