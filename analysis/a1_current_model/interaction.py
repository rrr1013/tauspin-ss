"""Teacher x world interaction on the H/Z AUC, paired bootstrap (review follow-up).

interaction = [AUC(cleo teacher) - AUC(gen teacher)]_CLEO world - [same]_gen world.
"""
import json

import numpy as np

import analyze as an
import reweight as rw


def main():
    d, nets = an.load_all()
    y, w0, w = d['y'], d['w0'], d['w']
    pairs = {'exact': (an.lr(d['h_gen']), an.lr(d['h_cleo']))}
    for arm in ('base_s43', 'full22_s42', 'full22_s43', 'idealip22_s42'):
        pairs[arm] = (nets[('gen', arm)]['score'], nets[('cleo', arm)]['score'])
    rng = np.random.default_rng(5)
    out = {}
    for p, m in {'any 3pi': d['is3'].any(1), '3pi x 3pi': d['is3'].all(1)}.items():
        idx = np.flatnonzero(m)
        draws = [rng.choice(idx, len(idx)) for _ in range(400)]
        for name, (sg, sc) in pairs.items():
            def inter(b):
                return ((an.wauc(sc[b], y[b], (w0 * w)[b]) - an.wauc(sg[b], y[b], (w0 * w)[b]))
                        - (an.wauc(sc[b], y[b], w0[b]) - an.wauc(sg[b], y[b], w0[b])))
            bb = [inter(b) for b in draws]
            out.setdefault(name, {})[p] = {'interaction': inter(idx), 'ci95': np.quantile(bb, [0.025, 0.975]).tolist()}
    (rw.HERE / 'results' / 'interaction.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
