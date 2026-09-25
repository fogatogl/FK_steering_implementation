/* The coalescence model of collapse_lab/n_coalescence.py, ported to JavaScript for the
   demo (W1 edge weights, W3 simulator). Given the normalised weights at each non-terminal
   scheduled step, it propagates the law of the root-per-slot vector exactly and returns the
   expected number of distinct roots at the end. Two resamplers: the systematic comb of
   smc/resampling.py integrated over its offset u ~ U(0, 1/k), applied at the steps where the
   run resampled (weights not flat, threshold 1.0); the multinomial draw, at every
   non-terminal step, flat weights included.

   Plain ES5 on purpose: the file is also run through an embedded engine in the test harness.
   Each function names the Python function it mirrors.
   Exact enumeration is k^k for the multinomial; the page uses it at k = 4 (the recorded
   regime) and falls back to a seeded Monte Carlo above (predictMC), and says which. */

var Coalescence = (function () {
  'use strict';

  // mirrors softmax(lg)
  function softmax(lg) {
    var mx = -Infinity, i, s = 0, w = [];
    for (i = 0; i < lg.length; i++) if (lg[i] > mx) mx = lg[i];
    for (i = 0; i < lg.length; i++) { w[i] = Math.exp(lg[i] - mx); s += w[i]; }
    for (i = 0; i < lg.length; i++) w[i] /= s;
    return w;
  }

  function ess(w) {
    var s = 0, i;
    for (i = 0; i < w.length; i++) s += w[i] * w[i];
    return 1 / s;
  }

  // numpy searchsorted(cumw, v, side='left'): first i with cumw[i] >= v
  function searchLeft(cumw, v) {
    var i;
    for (i = 0; i < cumw.length; i++) if (cumw[i] >= v) return i;
    return cumw.length - 1;
  }

  function cumsum1(w) {
    var c = [], s = 0, i;
    for (i = 0; i < w.length; i++) { s += w[i]; c[i] = s; }
    c[w.length - 1] = 1.0;          // cumw[-1] forced to 1, as smc/resampling.py does
    return c;
  }

  // mirrors affectations_peigne(w): [{p, idx}] the distinct assignments of the comb as u
  // sweeps [0, 1/k), each with the measure of u that produces it
  function combAssignments(w) {
    var k = w.length, cumw = cumsum1(w), bounds = [0.0, 1.0 / k], i, j, b, out = [], lo, hi, u, idx;
    for (i = 0; i < k - 1; i++) {
      for (j = 0; j < k; j++) {
        b = cumw[i] - j / k;
        if (b > 0.0 && b < 1.0 / k) bounds.push(b);
      }
    }
    bounds.sort(function (a, c) { return a - c; });
    for (i = 0; i + 1 < bounds.length; i++) {
      lo = bounds[i]; hi = bounds[i + 1];
      if (hi <= lo) continue;                  // duplicates of a bound (the Python uses a set)
      u = 0.5 * (lo + hi); idx = [];
      for (j = 0; j < k; j++) idx[j] = searchLeft(cumw, u + j / k);
      out.push({ p: (hi - lo) * k, idx: idx });
    }
    return out;
  }

  // mirrors affectations_multinomial(w): all k^k assignments with their probability
  function multinomialAssignments(w) {
    var k = w.length, n = Math.pow(k, k), out = [], a, rem, j, idx, p;
    for (a = 0; a < n; a++) {
      rem = a; idx = []; p = 1;
      for (j = 0; j < k; j++) { idx[j] = rem % k; rem = (rem - idx[j]) / k; p *= w[idx[j]]; }
      out.push({ p: p, idx: idx });
    }
    return out;
  }

  // mirrors propager(etats, affs); states: {"r0,r1,..": proba}; idx copies slot idx[j] into j
  function propagate(states, affs) {
    var next = {}, key, s, a, t, j, q;
    for (key in states) {
      if (!states.hasOwnProperty(key)) continue;
      s = key.split(',');
      for (a = 0; a < affs.length; a++) {
        q = affs[a].p;
        if (q < 1e-12) continue;
        t = [];
        for (j = 0; j < affs[a].idx.length; j++) t[j] = s[affs[a].idx[j]];
        t = t.join(',');
        next[t] = (next[t] || 0) + states[key] * q;
      }
    }
    return next;
  }

  function nDistinct(key) {
    var seen = {}, n = 0, s = key.split(','), i;
    for (i = 0; i < s.length; i++) if (!seen[s[i]]) { seen[s[i]] = true; n++; }
    return n;
  }

  // mirrors pas_du_run(r): from logG at every scheduled step (terminal included, unused),
  // the normalised weights at each non-terminal step and whether the run resampled there.
  // logW restarts from zero after a resampling. threshold 1.0: resample iff the increments
  // are not all equal; threshold < 1: ESS < threshold * k; threshold > 1: always.
  function stepsFromLogG(logG, threshold) {
    var k = logG[0].length, logW = [], out = [], m, j, w, e, res, mx, mn;
    for (j = 0; j < k; j++) logW[j] = 0;
    for (m = 0; m < logG.length - 1; m++) {
      for (j = 0; j < k; j++) logW[j] += logG[m][j];
      w = softmax(logW); e = ess(w);
      if (threshold === 1.0) {
        mx = -Infinity; mn = Infinity;
        for (j = 0; j < k; j++) { if (logG[m][j] > mx) mx = logG[m][j]; if (logG[m][j] < mn) mn = logG[m][j]; }
        res = (mx - mn) > 1e-9;
      } else if (threshold > 1.0) {
        res = true;
      } else {
        res = e < threshold * k;
      }
      out.push({ w: w, ess: e, res: res });
      if (res) for (j = 0; j < k; j++) logW[j] = 0;
    }
    return out;
  }

  // mirrors predire(r, resampler): steps from stepsFromLogG; {eDist, pOne, nRes}
  function predict(steps, k, resampler) {
    var init = [], j, states, m, affs, nRes = 0, key, eDist = 0, pOne = 0, d;
    for (j = 0; j < k; j++) init[j] = j;
    states = {}; states[init.join(',')] = 1.0;
    for (m = 0; m < steps.length; m++) {
      if (resampler === 'systematic') {
        if (!steps[m].res) continue;
        affs = combAssignments(steps[m].w);
      } else {
        affs = multinomialAssignments(steps[m].w);    // every step, flat weights included
      }
      nRes += 1;
      states = propagate(states, affs);
    }
    for (key in states) {
      if (!states.hasOwnProperty(key)) continue;
      d = nDistinct(key);
      eDist += states[key] * d;
      if (d === 1) pOne += states[key];
    }
    return { eDist: eDist, pOne: pOne, nRes: nRes };
  }

  /* The increments smc/fk.py writes for the `max` potential in increment form, rebuilt from
     the recorded guide rewards and the recorded parents (same rule as
     scripts/export_demo_data.py::logG_from_rewards): at step m > 0 and not terminal,
     lam * (max(r_m[j], M) - M) with M the running max of the parent anc[m-1][j]; at the
     first step lam * r; at the terminal step lam * (r - M). floor: r <- max(r, 0) first.
     The weights the resampler used are softmax of these accumulated (stepsFromLogG). */
  function logGFromRewards(r, anc, lam, floor) {
    var k = r[0].length, out = [], M = null, m, j, prev, curr, rm, last;
    for (m = 0; m < r.length; m++) {
      rm = [];
      for (j = 0; j < k; j++) rm[j] = floor ? Math.max(r[m][j], 0) : r[m][j];
      prev = []; curr = [];
      last = (m === r.length - 1);
      for (j = 0; j < k; j++) {
        prev[j] = (m === 0) ? 0 : M[anc[m - 1][j]];
        curr[j] = (last || m === 0) ? rm[j] : Math.max(rm[j], prev[j]);
      }
      out[m] = [];
      for (j = 0; j < k; j++) out[m][j] = lam * (curr[j] - prev[j]);
      M = curr;
    }
    return out;
  }

  // root of slot j after the last resampling, following the parents (matches root_slots)
  function rootsFromAncestors(anc) {
    var k = anc[0].length, roots = [], m, j, nxt;
    for (j = 0; j < k; j++) roots[j] = j;
    for (m = 0; m < anc.length - 1; m++) {     // the terminal row is the identity
      nxt = [];
      for (j = 0; j < k; j++) nxt[j] = roots[anc[m][j]];
      roots = nxt;
    }
    return roots;
  }

  // ---- one realisation, and Monte Carlo for k where k^k is out of reach ----------------
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // one comb draw, as resample_systematic: u ~ U(0, 1/k), points u + j/k, searchsorted left
  function drawSystematic(w, rng) {
    var k = w.length, cumw = cumsum1(w), u = rng() / k, idx = [], j;
    for (j = 0; j < k; j++) idx[j] = searchLeft(cumw, u + j / k);
    return idx;
  }

  function drawMultinomial(w, rng) {
    var k = w.length, cumw = cumsum1(w), idx = [], j;
    for (j = 0; j < k; j++) idx[j] = searchLeft(cumw, rng());
    return idx;
  }

  // one genealogy: the parents drawn at each non-terminal step (identity when no resampling)
  function simulateOne(steps, k, resampler, rng) {
    var anc = [], m, j, idx;
    for (m = 0; m < steps.length; m++) {
      if (resampler === 'systematic') {
        if (!steps[m].res) { idx = []; for (j = 0; j < k; j++) idx[j] = j; }
        else idx = drawSystematic(steps[m].w, rng);
      } else {
        idx = drawMultinomial(steps[m].w, rng);
      }
      anc.push(idx);
    }
    idx = []; for (j = 0; j < k; j++) idx[j] = j;
    anc.push(idx);                              // the terminal row, identity
    return anc;
  }

  function predictMC(steps, k, resampler, n, seed) {
    var rng = mulberry32(seed || 1), i, sum = 0, one = 0, roots, d;
    for (i = 0; i < n; i++) {
      roots = rootsFromAncestors(simulateOne(steps, k, resampler, rng));
      d = nDistinct(roots.join(','));
      sum += d; if (d === 1) one += 1;
    }
    return { eDist: sum / n, pOne: one / n, nRes: steps.length, n: n };
  }

  return {
    softmax: softmax, ess: ess, combAssignments: combAssignments,
    multinomialAssignments: multinomialAssignments, propagate: propagate,
    stepsFromLogG: stepsFromLogG, predict: predict, logGFromRewards: logGFromRewards,
    rootsFromAncestors: rootsFromAncestors, mulberry32: mulberry32,
    simulateOne: simulateOne, predictMC: predictMC, nDistinct: nDistinct
  };
})();
