#!/usr/bin/env python3
"""Train tiny 1D CNN for AAMI ECG beat classification on MIT-BIH.
Matches ecgnet.h exactly: 180-sample window, Conv1d k5 valid + ReLU + MaxPool2,
Linear 608->48 ReLU, Linear 48->5, optional logit bias. No BatchNorm/Dropout.
Protocol: AAMI EC57 DS1/DS2 patient-disjoint split; val = DS1 records 215/220/223/230.
Exports: weights.bin (raw float32, manifest order), beats_test.bin, rec200.bin, metrics.json.
"""
import numpy as np, struct, os, json
import torch, torch.nn as nn
import wfdb

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mitdb', 'mit-bih-arrhythmia-database-1.0.0')
OUT = os.path.dirname(os.path.abspath(__file__))
HALF = 90
DS1 = ['101','106','108','109','112','114','115','116','118','119','122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202','210','212','213','214','219','221','222','228','231','232','233','234']
AAMI = {'N':'N','L':'N','R':'N','e':'N','j':'N','A':'S','a':'S','J':'S','S':'S','V':'V','E':'V','F':'F','/':'Q','f':'Q','Q':'Q'}
CLS = ['N','S','V','F','Q']

def extract(records):
    beats, labels, recs = [], [], []
    for rec in records:
        ann = wfdb.rdann(os.path.join(DATADIR, rec), 'atr')
        sig = wfdb.rdrecord(os.path.join(DATADIR, rec)).p_signal[:, 0]
        for s, sym in zip(ann.sample, ann.symbol):
            if sym in AAMI and s >= HALF and s + HALF <= len(sig):
                w = sig[s-HALF:s+HALF].astype(np.float32)
                beats.append(w); labels.append(CLS.index(AAMI[sym])); recs.append(rec)
    X = np.stack(beats); y = np.array(labels); r = np.array(recs)
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-6
    return ((X - mu) / sd)[:, None, :], y, r

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv1d(1, 8, 5)
        s.c2 = nn.Conv1d(8, 16, 5)
        s.c3 = nn.Conv1d(16, 32, 5)
        s.f1 = nn.Linear(32 * 19, 48)
        s.f2 = nn.Linear(48, 5)
        s.pool = nn.MaxPool1d(2)
    def forward(s, x):
        x = s.pool(torch.relu(s.c1(x)))
        x = s.pool(torch.relu(s.c2(x)))
        x = s.pool(torch.relu(s.c3(x)))
        x = torch.relu(s.f1(x.flatten(1)))
        return s.f2(x)

def bacc(yt, yp):
    return float(np.mean([(((yp == c) & (yt == c)).sum().item() / max(1, (yt == c).sum().item())) for c in range(5)]))

def macro_f1(yt, yp):
    f1s = []
    for c in range(5):
        tp = ((yp == c) & (yt == c)).sum().item()
        fp = ((yp == c) & (yt != c)).sum().item()
        fn = ((yp != c) & (yt == c)).sum().item()
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1s))

def predict(model, X, bs=4096):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            preds.append(model(X[i:i+bs]).argmax(1))
    return torch.cat(preds)

def main():
    torch.manual_seed(7); np.random.seed(7)
    print('extracting DS1...', flush=True)
    X1, y1, r1 = extract(DS1)
    print('extracting DS2...', flush=True)
    X2, y2, r2 = extract(DS2)
    X1 = torch.tensor(X1); y1 = torch.tensor(y1)
    X2 = torch.tensor(X2); y2 = torch.tensor(y2)
    print('DS1:', np.bincount(y1.numpy()), 'DS2:', np.bincount(y2.numpy()), flush=True)
    counts = np.bincount(y1.numpy()).astype(np.float64)
    weights = np.minimum(counts.sum() / (5 * counts), 15.0)
    print('class weights (clip 15):', weights.round(2), flush=True)
    VAL_RECS = {'215', '220', '223', '230'}
    valm = np.array([r in VAL_RECS for r in r1])
    tr_idx = torch.tensor(np.nonzero(~valm)[0]); va_idx = torch.tensor(np.nonzero(valm)[0])
    print('train %d val %d (val patients %s, disjoint)' % (len(tr_idx), len(va_idx), sorted(VAL_RECS)), flush=True)
    model = Net()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    g = torch.Generator().manual_seed(123)
    best_va, best_state, patience = -1, None, 0
    for ep in range(60):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        tot = 0
        for i in range(0, len(perm), 1024):
            idx = perm[i:i+256]
            loss = lossf(model(X1[idx]), y1[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        vaP = predict(model, X1[va_idx])
        va_bal = bacc(y1[va_idx], vaP)
        va_acc = (vaP == y1[va_idx]).float().mean().item()
        print('epoch %d loss %.3f val %.4f valbal %.4f' % (ep, tot, va_acc, va_bal), flush=True)
        if va_bal > best_va:
            best_va, patience = va_bal, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 6: print('early stop', flush=True); break
    model.load_state_dict(best_state)
    # logit bias tuned on validation split only (fixed prior choice from val experiments)
    best_bvec = [0.5, 0.0, 0.0, 0.0, 0.0]
    for b, n in zip(model.f2.bias.data, best_bvec): b += n
    vaP = predict(model, X1[va_idx]); teP = predict(model, X2)
    va_acc = (vaP == y1[va_idx]).float().mean().item()
    te_acc = (teP == y2).float().mean().item()
    va_bal = bacc(y1[va_idx], vaP); te_bal = bacc(y2, teP)
    te_f1 = macro_f1(y2, teP)
    per = {}
    for c, name in enumerate(CLS):
        m = y2 == c
        per[name] = ((teP == c) & m).sum().item() / max(1, m.sum().item())
    maj = max(np.bincount(y2.numpy())) / len(y2)
    print('FINAL val %.4f valbal %.4f DS2 %.4f DS2bal %.4f DS2macroF1 %.4f majbase %.4f' % (va_acc, va_bal, te_acc, te_bal, te_f1, maj), flush=True)
    print('per-class DS2:', {k: round(v, 3) for k, v in per.items()}, flush=True)
    with open(os.path.join(OUT, 'metrics.json'), 'w') as f:
        json.dump({'val_acc': va_acc, 'val_balanced': va_bal, 'ds2_acc': te_acc, 'ds2_balanced': te_bal,
                   'ds2_macro_f1': te_f1, 'majority_baseline': maj, 'per_class_ds2': per,
                   'logit_bias': best_bvec,
                   'val_records': sorted(VAL_RECS), 'protocol': 'AAMI EC57 DS1/DS2 patient-disjoint',
                   'dataset': 'MIT-BIH Arrhythmia Database (PhysioNet, ODC-BY 1.0)'}, f, indent=2)
    # weights.bin: raw float32 LE, manifest order c1.w c1.b c2.w c2.b c3.w c3.b f1.w f1.b f2.w f2.b
    sd = model.state_dict()
    order = ['c1.weight','c1.bias','c2.weight','c2.bias','c3.weight','c3.bias','f1.weight','f1.bias','f2.weight','f2.bias']
    with open(os.path.join(OUT, 'weights.bin'), 'wb') as f:
        for k in order:
            t = sd[k].detach().numpy().astype(np.float32)
            # undo the logit-bias tweak for f2.bias export: bias applied separately in C
            if k == 'f2.bias':
                t = t - np.array(best_bvec, dtype=np.float32)
            f.write(t.reshape(-1).tobytes())
    print('weights.bin %d bytes' % os.path.getsize(os.path.join(OUT, 'weights.bin')), flush=True)
    # beats_test.bin: 5000 DS2 beats (u32 n; per beat u8 label + 180 float32)
    rng = np.random.default_rng(11)
    sel = rng.choice(len(X2), size=min(5000, len(X2)), replace=False)
    with open(os.path.join(OUT, 'beats_test.bin'), 'wb') as f:
        f.write(struct.pack('<I', len(sel)))
        Xn = X2.numpy(); yn = y2.numpy()
        for i in sel:
            f.write(struct.pack('<B', int(yn[i])))
            f.write(Xn[i, 0].astype('<f4').tobytes())
    # rec200.bin: full signal + beat positions/labels for the demo (u32 nsig; float sig; u32 nbeats; u32 pos[]; u8 lab[])
    rec = '200'
    ann = wfdb.rdann(os.path.join(DATADIR, rec), 'atr')
    sig = wfdb.rdrecord(os.path.join(DATADIR, rec)).p_signal[:, 0].astype(np.float32)
    pos, lab = [], []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym in AAMI and s >= HALF and s + HALF <= len(sig):
            pos.append(int(s)); lab.append(CLS.index(AAMI[sym]))
    with open(os.path.join(OUT, 'rec200.bin'), 'wb') as f:
        f.write(struct.pack('<I', len(sig))); f.write(sig.astype('<f4').tobytes())
        f.write(struct.pack('<I', len(pos)))
        f.write(np.array(pos, dtype='<u4').tobytes())
        f.write(bytes(lab))
    print('beats_test.bin + rec200.bin written (%d beats)' % len(pos), flush=True)

if __name__ == '__main__':
    main()
