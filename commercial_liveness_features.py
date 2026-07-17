"""Float32 Python port of the commercial liveness PPG and g_acc features."""

import math

import numpy as np


F32 = np.float32
HALF = 125
LD_BUF = 125
MAX_PEAK = 16
FFTLEN = 256
PI_FFT = F32(3.141592653589793)
PI_FT = F32(3.14159274)
SQ2 = F32(0.707106769)
CM = F32(0.01)

ZI = np.array([
    -2.19935876743021,
    -2.19935876743197,
    4.39871753486313,
    4.39871753485981,
    -2.19935876743003,
    -2.19935876743073,
], dtype=F32)
AC = np.array([
    100.0,
    -424.063910733117,
    779.557874551658,
    -801.521631217565,
    488.518691595902,
    -167.170254015974,
    24.9771551375827,
], dtype=F32)
BC = np.array([
    2.19935876743062,
    0.0,
    -6.59807630229186,
    0.0,
    6.59807630229186,
    0.0,
    -2.19935876743062,
], dtype=F32)


def _median5(arr):
    """Five-point median filter with C-compatible zero padding."""
    n = len(arr)
    out = np.zeros(n, dtype=F32)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        win = np.zeros(5, dtype=F32)
        if i < 2:
            win[2 - i:] = arr[lo:hi]
        elif i >= n - 2:
            win[:hi - lo] = arr[lo:hi]
        else:
            win = arr[lo:hi]
        out[i] = F32(np.sort(win)[2])
    return out


def _iir_filter(x, zi):
    """Direct-form IIR with the same in-place update order as the C filter."""
    fa = np.zeros(7, dtype=F32)
    fa[1:] = zi
    out = np.zeros(len(x), dtype=F32)
    for j in range(len(x)):
        fa[:6] = fa[1:7]
        fa[6] = F32(0)
        for i in range(7):
            fa[i] = F32(fa[i] + x[j] * BC[i] * CM)
        for i in range(6):
            fa[i + 1] = F32(fa[i + 1] - fa[0] * AC[i + 1] * CM)
        out[j] = fa[0]
    return out


def _bpfiltfilt(x):
    """Zero-phase band-pass filter with an 18-sample reflected pad."""
    p = 18
    n = len(x)
    if n <= p * 2:
        return x.copy()
    head = F32(F32(2.0) * x[0])
    end = F32(F32(2.0) * x[n - 1])
    ff = np.empty(n + 2 * p, dtype=F32)
    ff[:p] = head - x[p:0:-1]
    ff[p:p + n] = x
    ff[p + n:] = end - x[n - 2:n - 2 - p:-1]
    z1 = F32(ZI * ff[0] * CM)
    ff = _iir_filter(ff, z1)
    ff = ff[::-1]
    z2 = F32(ZI * ff[0] * CM)
    ff = _iir_filter(ff, z2)
    ff = ff[::-1]
    return ff[p:p + n].copy()


def _moving_avg(x):
    n = len(x)
    out = np.zeros(n, dtype=F32)
    mb = np.zeros(5, dtype=F32)
    for i in range(n):
        mb[:4] = mb[1:5]
        mb[4] = x[i]
        out[i] = F32(np.mean(mb))
    return out


def _fft_four1(data, nn, isign=1):
    """Numerical Recipes four1 using one-indexed interleaved complex data."""
    d = data.copy()
    n = nn << 1
    j = 1
    for i in range(1, n, 2):
        if j > i:
            d[j], d[i] = d[i], d[j]
            d[j + 1], d[i + 1] = d[i + 1], d[j + 1]
        m = nn
        while m >= 2 and j > m:
            j -= m
            m >>= 1
        j += m
    mmax = 2
    while n > mmax:
        istep = mmax << 1
        theta = F32(isign * (F32(6.28318530717959) / F32(mmax)))
        wtemp = F32(np.sin(F32(0.5) * theta))
        wpr = F32(-2.0 * wtemp * wtemp)
        wpi = F32(np.sin(theta))
        wr = F32(1.0)
        wi = F32(0.0)
        for m in range(1, mmax, 2):
            i = m
            while i <= n:
                jj = i + mmax
                tr = F32(wr * d[jj] - wi * d[jj + 1])
                ti = F32(wr * d[jj + 1] + wi * d[jj])
                d[jj] = F32(d[i] - tr)
                d[jj + 1] = F32(d[i + 1] - ti)
                d[i] = F32(d[i] + tr)
                d[i + 1] = F32(d[i + 1] + ti)
                i += istep
            wtemp2 = wr
            wr = F32(wtemp2 * wpr - wi * wpi + wr)
            wi = F32(wi * wpr + wtemp2 * wpi + wi)
        mmax = istep
    return d


def _fft_handle(buf):
    """Hann window, 256-point FFT, and C-compatible power spectrum."""
    n = len(buf)
    for i in range(n):
        t = F32(F32(0.5) - F32(0.5 * np.cos(
            F32(2.0 * PI_FFT * F32(i + 1) / F32(n + 1))
        )))
        buf[i] = F32(buf[i] * t)
    ft = np.zeros(FFTLEN * 2 + 1, dtype=F32)
    for i in range(FFTLEN):
        if i < n:
            ft[2 * i + 1] = buf[i]
    ft = _fft_four1(ft, FFTLEN, 1)
    out = np.zeros(250, dtype=F32)
    for i in range(250):
        out[i] = F32(
            np.power(ft[2 * i + 1], 2)
            + np.power(ft[2 * (i + 1)], 2)
        )
    return out


def _find_peak_valley(buf):
    """C-compatible FindPeaks and FindValleys implementation."""
    n = len(buf)
    maxp = MAX_PEAK
    mind = int(7.5 / 2 + 0.5)
    minh = F32(0)
    gr = F32(0.1)
    pt = np.zeros(maxp, dtype=np.int16)
    pv = np.zeros(maxp, dtype=F32)
    vt = np.zeros(maxp, dtype=np.int16)
    vv = np.zeros(maxp, dtype=F32)
    pn = vn = 0
    presign = 0
    pretime = 0
    prepole = F32(0)

    def addp(k):
        nonlocal pn, presign, pretime, prepole
        pn += 1
        if pn > maxp:
            pn = maxp
            pt[:maxp - 1] = pt[1:maxp]
            pv[:maxp - 1] = pv[1:maxp]
        pt[pn - 1] = k
        pv[pn - 1] = buf[k]

    def addv(k):
        nonlocal vn, presign, pretime, prepole
        vn += 1
        if vn > maxp:
            vn = maxp
            vt[:maxp - 1] = vt[1:maxp]
            vv[:maxp - 1] = vv[1:maxp]
        vt[vn - 1] = k
        vv[vn - 1] = buf[k]

    for k in range(1, n - 1):
        kk = max(k, 1)
        cont = False
        if buf[kk] > buf[kk - 1] and buf[kk] >= buf[kk + 1]:
            if pretime == 0:
                presign = 1
                pretime = kk
                prepole = buf[kk]
                addp(kk)
                cont = True
            elif kk - pretime < mind:
                cont = True
            elif presign != 1 and buf[kk] - prepole >= minh:
                if pn >= 1:
                    dpre = float(pv[pn - 1]) - float(vv[vn - 1])
                    dh = float(buf[kk]) - float(vv[vn - 1])
                    if dh > dpre * float(gr):
                        presign = 1
                        pretime = kk
                        prepole = buf[kk]
                        addp(kk)
            elif presign == 1 and buf[kk] > prepole:
                pretime = kk
                prepole = buf[kk]
                pt[pn - 1] = kk
                pv[pn - 1] = buf[kk]
        if cont:
            continue
        if presign != 0 and buf[kk] <= buf[kk - 1] and buf[kk] < buf[kk + 1]:
            if kk - pretime < mind:
                continue
            if presign != -1 and prepole - buf[kk] >= minh:
                presign = -1
                pretime = kk
                prepole = buf[kk]
                addv(kk)
            elif presign == -1 and buf[kk] < prepole:
                pretime = kk
                prepole = buf[kk]
                vt[vn - 1] = kk
                vv[vn - 1] = buf[kk]
    return (
        pn,
        pt[:pn].copy(),
        pv[:pn].copy(),
        vn,
        vt[:vn].copy(),
        vv[:vn].copy(),
    )


def _cal_ac(buf):
    pn, pt, pv, vn, vt, vv = _find_peak_valley(buf)
    mn = min(pn, vn)
    if mn == 0:
        return F32(0), pn, pt, pv, vn, vt, vv
    ac = F32(pv[:mn] - vv[:mn])
    return _median_sort(ac), pn, pt, pv, vn, vt, vv


def _median_sort(arr):
    n = len(arr)
    if n < 2:
        return F32(0)
    a = arr.copy()
    for i in range(1, n):
        t = a[i]
        si = i - 1
        while si >= 0 and a[si] > t:
            a[si + 1] = a[si]
            si -= 1
        a[si + 1] = t
    if n % 2 == 0:
        return F32((a[n // 2 - 1] + a[n // 2]) / F32(2.0))
    return F32(a[(n - 1) // 2])


def _std(a):
    n = len(a)
    if n <= 1:
        return F32(0)
    m = F32(np.mean(a))
    s = F32(0)
    for v in a:
        d = F32(v - m)
        s = F32(s + d * d)
    return F32(np.sqrt(F32(s / F32(n - 1))))


def _corrcoef(b1, b2):
    n = len(b1)
    if n <= 1:
        return F32(0)
    m1 = F32(np.mean(b1))
    m2 = F32(np.mean(b2))
    s1 = _std(b1)
    s2 = _std(b2)
    if s1 == 0 or s2 == 0:
        return F32(0)
    r = F32(0)
    for i in range(n):
        r = F32(r + F32(F32(b1[i] - m1) * F32(b2[i] - m2) / s1 / s2))
    return F32(r / F32(n - 1))


def _downsample(src, dst_len):
    sl = len(src)
    dst = list(src.astype(F32))
    rn = sl - dst_len
    if rn == 0:
        return np.array(dst, dtype=F32)
    step = F32(F32(sl) / F32(rn + 1))
    ln = sl
    for i in range(1, rn + 1):
        ri = int(math.ceil(step * i)) - 1 - (i - 1)
        ln -= 1
        for ii in range(ri, ln):
            dst[ii] = dst[ii + 1]
    return np.array(dst[:dst_len], dtype=F32)


def _pattern_corr(buf, vt, si, cn):
    pl = int(vt[si + 1] - vt[si] + 1)
    if pl > HALF:
        return F32(0)
    pat = F32([buf[int(vt[si]) + i] for i in range(pl)])
    ct = np.zeros(MAX_PEAK, dtype=F32)
    num = 0
    for idx in range(cn):
        tl = int(vt[idx + 2 + si] - vt[idx + 1 + si] + 1)
        if tl > HALF:
            ct[num] = F32(0)
            num += 1
            continue
        tcs = F32([buf[int(vt[idx + 1 + si]) + j] for j in range(tl)])
        if pl >= tl:
            comp = _downsample(pat, tl)
            tc = _corrcoef(tcs, comp)
        else:
            comp = _downsample(tcs, pl)
            tc = _corrcoef(pat, comp)
        ct[num] = tc
        num += 1
    ct[:num] = F32(np.abs(ct[:num]))
    return _median_sort(ct[:num].copy())


def _get_corrcoef(buf, vt, vn):
    if vn == 0:
        return F32(0)
    beat = vn - 1
    if beat < 2 or beat > 16:
        return F32(0)
    c1 = _pattern_corr(buf, vt, 0, beat - 1)
    c2 = F32(0)
    if beat >= 4:
        c2 = _pattern_corr(buf, vt, 1, beat - 2)
    return F32(c1 if c1 > c2 else c2)


def _xcorr125(buf):
    tb = np.zeros(HALF, dtype=F32)
    for i in range(HALF):
        s = F32(0)
        for j in range(i + 1):
            s = F32(s + buf[j] * buf[HALF - 1 - i + j])
        tb[i] = s
    buf[:] = tb
    return buf


def _fft_feature(buf):
    hr = _fft_handle(buf)
    for i in range(FFTLEN // 2):
        hr[i] = F32(F32(hr[i] / F32(F32(25) * FFTLEN)) * F32(2))
    cut = F32([hr[5 + i - 1] for i in range(34)])
    mx = F32(np.max(cut))
    med = _median_sort(cut.copy())
    return F32(0) if med == 0 else F32(mx / med)


def _remove_burr(buf, th=500):
    for i in range(len(buf) - 2):
        d1 = F32(buf[i + 1] - buf[i])
        d2 = F32(buf[i + 1] - buf[i + 2])
        if abs(d1) > th and abs(d2) > th and d1 * d2 > 0:
            buf[i + 1] = buf[i]
    for i in range(len(buf) - 3):
        d1 = F32(buf[i + 1] - buf[i])
        d2 = F32(buf[i + 2] - buf[i + 3])
        if abs(d1) > th and abs(d2) > th and d1 * d2 > 0:
            buf[i + 1] = buf[i]
            buf[i + 2] = buf[i]
    return buf


def _remove_step(buf, th=50000):
    sv = []
    si = []
    i = 0
    while i < len(buf) - 1 and len(sv) < 10:
        if abs(F32(buf[i + 1] - buf[i])) > th:
            sv.append(F32(buf[i + 1] - buf[i]))
            si.append(i)
        i += 1
    for s in range(len(sv)):
        for j in range(si[s] + 1):
            buf[j] = F32(buf[j] + sv[s])
    return buf


def _preprocess(raw):
    buf = F32([float(v) for v in raw])
    buf = _remove_burr(buf)
    buf = _remove_step(buf)
    dc = F32(abs(np.mean(buf)))
    flt = _moving_avg(_bpfiltfilt(_median5(buf)))
    return flt, dc


F_NAMES = [
    "green_corr",
    "green_ac",
    "amb_ac",
    "acc_ysum",
    "green_dc",
    "amb_dc",
    "green_xcorr",
    "fft_peak_med",
]


def cal_ppg_feature(ppg_raw, g_acc):
    """Return the eight C-compatible float32 features for a (125, 4) PPG buffer."""
    ff = np.zeros(8, dtype=F32)
    ff[3] = g_acc
    fb_amb, dc_amb = _preprocess(ppg_raw[:, 3])
    ff[5] = dc_amb
    ff[2], _, _, _, _, _, _ = _cal_ac(fb_amb)
    fb_g, dc_g = _preprocess(ppg_raw[:, 0])
    ff[4] = dc_g
    ff[1], _, _, _, vn, vt, _ = _cal_ac(fb_g)
    ff[0] = _get_corrcoef(fb_g, vt, vn)
    xc = _xcorr125(fb_g.copy())
    _, _, _, _, vn2, vt2, _ = _cal_ac(xc)
    ff[6] = _get_corrcoef(xc, vt2, vn2)
    ff[7] = _fft_feature(xc)
    return ff


ACC_CH = 3
ACC_NOTCH = 5
FIFO_LEN = 50


class _AccState:
    def __init__(self):
        self.sp = [
            {
                "y0": F32(0),
                "y": np.zeros(ACC_CH, dtype=F32),
                "dAvg": F32(0),
            }
            for _ in range(ACC_CH)
        ]
        self.bp = [
            {
                "u": np.zeros(ACC_NOTCH, dtype=F32),
                "a": F32(0),
                "d1": F32(0),
                "d2": F32(0),
                "d3": F32(0),
                "d4": F32(0),
            }
            for _ in range(ACC_CH)
        ]
        self.u = np.zeros(FIFO_LEN, dtype=F32)
        self.ySum = F32(0)
        self.isStill = 0


def _spike_step(x, st, dAvgMin=F32(0.005), diffMax=F32(1e10)):
    y = st["y"]
    y[2] = y[1]
    y[1] = y[0]
    y[0] = F32(x - st["y0"])
    t1 = F32(y[1] - y[0])
    t2 = F32(y[2] - y[1])
    if t1 * t2 < 0 and abs(F32(t2 / st["dAvg"])) > 10 and abs(F32(t1 / st["dAvg"])) > 10:
        st["y"][1] = F32((y[0] + y[2]) / F32(2.0))
        return st["y"][1]
    if abs(t2) > diffMax:
        return st["y"][1]
    if t1 > F32(2.22044605E-15):
        st["dAvg"] = F32(F32(0.99) * st["dAvg"] + F32(0.01) * t1)
    if st["dAvg"] < dAvgMin:
        st["dAvg"] = dAvgMin
    return st["y"][1]


def _bp_step(x, st):
    u = st["u"]
    u[0] = F32(
        F32(
            F32(F32(st["d1"] * u[1]) + F32(st["d2"] * u[2]))
            + F32(st["d3"] * u[3])
        )
        + F32(st["d4"] * u[4])
        + x
    )
    y = F32(st["a"] * F32(F32(u[0] - F32(2.0) * u[2]) + u[4]))
    u[4] = u[3]
    u[3] = u[2]
    u[2] = u[1]
    u[1] = u[0]
    return y


def compute_g_acc(acc_xyz):
    """Return the C-compatible g_acc value for raw accelerometer counts."""
    st = _AccState()
    a0 = [
        F32(acc_xyz[0][0] / 4096.0),
        F32(acc_xyz[0][1] / 4096.0),
        F32(acc_xyz[0][2] / 4096.0),
    ]
    for k in range(3):
        st.sp[k] = {
            "y0": F32(a0[k]),
            "y": np.zeros(3, dtype=F32),
            "dAvg": F32(0.005),
        }
    fSa = F32(25)
    fBp = F32(min(20.0, fSa / 3.0))
    a = F32(
        np.cos(PI_FT * (fBp + F32(0.5)) / fSa)
        / np.cos(PI_FT * (fBp - F32(0.5)) / fSa)
    )
    x = F32(np.tan(PI_FT * (fBp - F32(0.5)) / fSa))
    b2 = F32(x * x)
    c = F32(b2 + F32(2.0) * x * SQ2)
    c1 = F32(c + F32(1.0))
    for j in range(3):
        st.bp[j]["a"] = F32(b2 / c1)
        st.bp[j]["d1"] = F32(F32(4.0) * a * F32(F32(1.0) + x * SQ2) / c1)
        st.bp[j]["d2"] = F32(F32(2.0) * (F32(b2 - F32(2.0) * a * a) - F32(1.0)) / c1)
        st.bp[j]["d3"] = F32(F32(4.0) * a * F32(F32(1.0) - x * SQ2) / c1)
        st.bp[j]["d4"] = F32(-F32(b2 - F32(2.0) * x * SQ2 + F32(1.0)) / c1)
    g_acc = F32(0)
    for i in range(1, len(acc_xyz)):
        a = [
            F32(acc_xyz[i][0] / 4096.0),
            F32(acc_xyz[i][1] / 4096.0),
            F32(acc_xyz[i][2] / 4096.0),
        ]
        bp = [F32(0), F32(0), F32(0)]
        for k in range(3):
            bp[k] = _bp_step(_spike_step(a[k], st.sp[k]), st.bp[k])
        y = F32(0)
        for k in range(3):
            v = F32(abs(bp[k]))
            bp[k] = F32(v * v)
            y = F32(y + bp[k])
        st.ySum = F32(st.ySum - st.u[49] + y)
        st.u[1:] = st.u[:-1]
        st.u[0] = y
        ys50 = F32(st.ySum / F32(50.0))
        isStill = 1 if ys50 < F32(0.005) else 0
        if st.isStill <= 1:
            g_acc = ys50
        st.isStill = isStill
    return g_acc


def main(ppg, acc):
    """Return eight float32 features from `(125, 4)` PPG and raw-count ACC."""
    ppg = np.asarray(ppg)
    acc = np.asarray(acc)
    g_acc = compute_g_acc(acc)
    return cal_ppg_feature(ppg, g_acc)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        p = np.fromfile(sys.argv[1], dtype=np.int32).reshape(125, 4)
        a = np.fromfile(sys.argv[2], dtype=np.int16).reshape(-1, 3)
    else:
        print("usage: python commercial_liveness_features.py <ppg.bin> <acc.bin>")
        sys.exit(2)
    feat = main(p, a)
    for name, value in zip(F_NAMES, feat):
        print(f"{name} = {float(value):.6g}")
