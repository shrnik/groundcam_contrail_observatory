(function () {
  const BIN_MIN  = 30;  // minutes per time bin
  const FL_STEP  = 10;  // flight level bin size

  // PuOr_r: t=0 → dark purple (clear), t=0.5 → white (neutral), t=1 → dark orange (contrail)
  const PUOR_R = [
    [45,  0,  75],   // 0.0
    [84,  39, 136],  // 0.1
    [128, 115, 172], // 0.2
    [178, 171, 210], // 0.3
    [216, 218, 235], // 0.4
    [247, 247, 247], // 0.5 — neutral
    [254, 224, 182], // 0.6
    [253, 184,  99], // 0.7
    [224, 130,  20], // 0.8
    [179,  88,   6], // 0.9
    [127,  59,   8], // 1.0
  ];

  function lerpColor(t) {
    t = Math.max(0, Math.min(1, t));
    const pos = t * (PUOR_R.length - 1);
    const lo  = Math.floor(pos);
    const hi  = Math.min(lo + 1, PUOR_R.length - 1);
    const f   = pos - lo;
    const c   = PUOR_R[lo].map((v, i) => Math.round(v + f * (PUOR_R[hi][i] - v)));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  function altToFL(altM) {
    // FL = hundreds of feet, binned to nearest FL_STEP
    return Math.round((altM * 3.28084) / (FL_STEP * 100)) * FL_STEP;
  }

  const BIN_MS = BIN_MIN * 60 * 1000;

  function floorBin(ms) { return Math.floor(ms / BIN_MS) * BIN_MS; }
  function ceilBin(ms)  { return Math.ceil(ms  / BIN_MS) * BIN_MS; }

  function toUTCHHMM(ms) {
    return new Intl.DateTimeFormat('en-US', {
      timeZone: 'UTC',
      hour: '2-digit', minute: '2-digit', hour12: false,
    }).format(new Date(ms));
  }

  function parseUTC(ts) {
    // ts may be "YYYY-MM-DD HH:MM:SS" (no Z)
    return new Date(
      ts.includes('Z') || ts.includes('+') ? ts : ts.replace(' ', 'T') + 'Z'
    ).getTime();
  }

  function renderHeatmap(frames, container) {
    container.innerHTML = '';

    // Flatten frames → per-aircraft observations
    const obs = [];
    for (const frame of frames) {
      const tMs = parseUTC(frame.timestamp);
      if (!isFinite(tMs)) continue;
      for (const ac of frame.aircraft || []) {
        const altM = parseFloat(ac.alt_m ?? 0);
        if (!altM) continue;
        obs.push({ t: tMs, fl: altToFL(altM), contrail: ac.contrail === '1' });
      }
    }

    if (obs.length === 0) {
      container.innerHTML = '<div style="color:#8b949e;padding:32px;text-align:center">No aircraft data for heatmap.</div>';
      return;
    }

    // Time bins
    const minT = Math.min(...obs.map(o => o.t));
    const maxT = Math.max(...obs.map(o => o.t));
    const startT = floorBin(minT);
    const endT   = ceilBin(maxT + 1);
    const timeBins = [];
    for (let t = startT; t < endT; t += BIN_MS) timeBins.push(t);
    const nT = timeBins.length;

    // FL bins (ascending)
    const allFL = obs.map(o => o.fl);
    const flMin = Math.min(...allFL);
    const flMax = Math.max(...allFL);
    const flBins = [];
    for (let fl = flMin; fl <= flMax; fl += FL_STEP) flBins.push(fl);
    const nFL = flBins.length;
    const flMap = Object.fromEntries(flBins.map((fl, i) => [fl, i]));

    // Count contrails and clears per cell separately
    const contrailCounts = Array.from({ length: nFL }, () => new Uint16Array(nT));
    const clearCounts    = Array.from({ length: nFL }, () => new Uint16Array(nT));
    for (const o of obs) {
      const ti = Math.floor((o.t - startT) / BIN_MS);
      const fi = flMap[o.fl] ?? -1;
      if (ti >= 0 && ti < nT && fi >= 0) {
        if (o.contrail) contrailCounts[fi][ti]++;
        else            clearCounts[fi][ti]++;
      }
    }

    // Score = contrail_count * CONTRAIL_WEIGHT - clear_count
    // Higher weight ensures even a single contrail sighting pushes the bin orange
    const CONTRAIL_WEIGHT = 20;
    const scores = Array.from({ length: nFL }, () => new Float32Array(nT));
    let vmax = 1;
    for (let fi = 0; fi < nFL; fi++) {
      for (let ti = 0; ti < nT; ti++) {
        const score = contrailCounts[fi][ti] * CONTRAIL_WEIGHT - clearCounts[fi][ti];
        scores[fi][ti] = score;
        if (Math.abs(score) > vmax) vmax = Math.abs(score);
      }
    }

    // Layout
    const PAD = { top: 24, right: 20, bottom: 46, left: 56 };
    const COLORBAR_H = 14;
    const LABEL_FONT = '11px ui-monospace,monospace';
    const availW = (container.clientWidth || 900) - PAD.left - PAD.right;
    const cellW  = Math.max(4, Math.min(36, Math.floor(availW / nT)));
    const cellH  = Math.max(8, Math.min(22, Math.floor(280 / nFL)));
    const W = PAD.left + cellW * nT + PAD.right;
    const H = PAD.top  + cellH * nFL + PAD.bottom;

    const canvas = document.createElement('canvas');
    canvas.width  = W;
    canvas.height = H;
    canvas.style.cssText = `display:block;width:100%;max-width:${W}px;image-rendering:pixelated`;

    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#161b22';
    ctx.fillRect(0, 0, W, H);

    // Cells — fl index 0 = lowest FL, draw at bottom
    for (let fi = 0; fi < nFL; fi++) {
      for (let ti = 0; ti < nT; ti++) {
        const score = scores[fi][ti];
        const t = (score + vmax) / (2 * vmax);  // 0=purple, 0.5=white, 1=orange
        ctx.fillStyle = lerpColor(t);
        const x = PAD.left + ti * cellW;
        const y = PAD.top  + (nFL - 1 - fi) * cellH;  // high FL at top
        ctx.fillRect(x, y, cellW - 1, cellH - 1);
      }
    }

    ctx.font      = LABEL_FONT;
    ctx.fillStyle = '#8b949e';

    // Y axis: FL labels every 2 bins
    ctx.textAlign    = 'right';
    ctx.textBaseline = 'middle';
    for (let fi = 0; fi < nFL; fi += 2) {
      const y = PAD.top + (nFL - 1 - fi) * cellH + cellH / 2;
      ctx.fillText(`FL${flBins[fi]}`, PAD.left - 4, y);
    }

    // X axis: one label per hour
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'top';
    const tickEvery  = Math.max(1, Math.round(60 / BIN_MIN));
    for (let ti = 0; ti < nT; ti += tickEvery) {
      const x = PAD.left + ti * cellW + cellW / 2;
      ctx.fillText(toUTCHHMM(timeBins[ti]), x, PAD.top + cellH * nFL + 6);
    }

    // X axis label
    ctx.textAlign = 'center';
    ctx.fillText(`Time (UTC, ${BIN_MIN}min bins)`, PAD.left + (cellW * nT) / 2, PAD.top + cellH * nFL + 20);

    // Colorbar
    const barX = PAD.left;
    const barY = PAD.top + cellH * nFL + 36;
    const barW = cellW * nT;
    for (let i = 0; i < barW; i++) {
      ctx.fillStyle = lerpColor(i / barW);
      ctx.fillRect(barX + i, barY, 1, COLORBAR_H);
    }
    ctx.strokeStyle = '#30363d';
    ctx.strokeRect(barX, barY, barW, COLORBAR_H);

    ctx.textAlign    = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle    = '#8073ac';
    ctx.fillText('◀ clear', barX, barY + COLORBAR_H + 2);

    ctx.textAlign = 'right';
    ctx.fillStyle = '#e08214';
    ctx.fillText('contrail ▶', barX + barW, barY + COLORBAR_H + 2);

    container.appendChild(canvas);
  }

  window.renderHeatmap = renderHeatmap;
})();
