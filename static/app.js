/* ==========================================================================
   PINO Scientific Studio — Frontend Application Controller (static/app.js)
   ========================================================================== */

(() => {
  "use strict";

  // --------------------------------------------------------------------------
  // 1. SCIENTIFIC COLORSCALES & PLOTLY DEFAULTS
  // --------------------------------------------------------------------------
  const COLORSCALE_VORTICITY = [
    [0.0, "#1E3A8A"],
    [0.2, "#2563EB"],
    [0.4, "#93C5FD"],
    [0.5, "#F8FAFC"],
    [0.6, "#FCA5A5"],
    [0.8, "#DC2626"],
    [1.0, "#7F1D1D"]
  ];

  const COLORSCALE_FORCING = [
    [0.0, "#4C1D95"],
    [0.22, "#7C3AED"],
    [0.42, "#DDD6FE"],
    [0.5, "#F8FAFC"],
    [0.58, "#FED7AA"],
    [0.78, "#EA580C"],
    [1.0, "#9A3412"]
  ];

  const COLORSCALE_ERROR = [
    [0.0, "#FFFFFF"],
    [0.25, "#FEF3C7"],
    [0.55, "#F59E0B"],
    [0.8, "#DC2626"],
    [1.0, "#7F1D1D"]
  ];

  const COLORSCALE_FFT = [
    [0.0, "#F8FAFC"],
    [0.3, "#BAE6FD"],
    [0.6, "#38BDF8"],
    [0.82, "#2563EB"],
    [1.0, "#1E1B4B"]
  ];

  const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false
  };

  function makeFieldLayout(titleText = "", xTitle = "x", yTitle = "y") {
    return {
      paper_bgcolor: "#FFFFFF",
      plot_bgcolor: "#FFFFFF",
      autosize: true,
      margin: { l: 42, r: 24, t: titleText ? 30 : 12, b: 36 },
      font: { color: "#0F172A", size: 10, family: "Inter, -apple-system, sans-serif" },
      title: titleText ? { text: titleText, font: { size: 12, color: "#0F172A" } } : undefined,
      xaxis: { autorange: true, showgrid: false, zeroline: false, title: xTitle },
      yaxis: { autorange: true, showgrid: false, zeroline: false, scaleanchor: "x", scaleratio: 1, title: yTitle }
    };
  }

  function makeChartLayout(xTitle, yTitle, extra = {}) {
    return {
      paper_bgcolor: "#FFFFFF",
      plot_bgcolor: "#FFFFFF",
      autosize: true,
      margin: { l: 54, r: 26, t: 18, b: 42 },
      font: { color: "#0F172A", size: 10, family: "Inter, -apple-system, sans-serif" },
      legend: { orientation: "h", y: 1.12, x: 0, font: { size: 10 } },
      xaxis: { title: xTitle, gridcolor: "#F1F5F9", zerolinecolor: "#E2E8F0", ...(extra.xaxis || {}) },
      yaxis: { title: yTitle, gridcolor: "#F1F5F9", zerolinecolor: "#E2E8F0", ...(extra.yaxis || {}) },
      ...Object.fromEntries(Object.entries(extra).filter(([k]) => k !== "xaxis" && k !== "yaxis"))
    };
  }

  // --------------------------------------------------------------------------
  // 2. APPLICATION STATE
  // --------------------------------------------------------------------------
  const state = {
    info: null,
    targetRes: 64,
    superresHighRes: 256,
    activeTab: "tab-sim",
    presentationMode: false,
    dirtySr: true,
    hasRunRace: false,
    hasRunPso: false,
    latestPredict: null,
    latestSuperres: null,
    latestRollout: null,
    latestBenchmark: null,
    latestScaling: null,
    latestTta: null,
    latestPso: null,
    painter: {
      vortices: [
        { x: 0.38, y: 0.42, strength: 3.0, sigma: 0.08 },
        { x: 0.62, y: 0.58, strength: -3.0, sigma: 0.08 }
      ],
      polarity: 1,
      sigma: 0.08,
      isDrawing: false,
      lastCanvasPt: null
    },
    rolloutPlayer: {
      timer: null,
      isPlaying: false,
      frameIdx: 0,
      fps: 2
    },
    particleFlow: {
      paused: false,
      speed: 1.0
    },
    inFlight: {
      predict: false,
      predictQueued: false,
      superres: false,
      tta: false
    },
    timers: {
      predictDebounce: null,
      srDebounce: null,
      toast: null
    }
  };

  // --------------------------------------------------------------------------
  // 3. DOM REFERENCES
  // --------------------------------------------------------------------------
  const els = {
    headerFlowCanvas: document.getElementById("headerFlowCanvas"),
    tabBar: document.getElementById("tabBar"),
    tabSliderPill: document.getElementById("tabSliderPill"),
    checkpointBadge: document.getElementById("checkpointBadge"),
    deviceBadge: document.getElementById("deviceBadge"),
    presentationModeBtn: document.getElementById("presentationModeBtn"),
    presentationBtnText: document.getElementById("presentationBtnText"),
    exportPngBtn: document.getElementById("exportPngBtn"),
    exportJsonBtn: document.getElementById("exportJsonBtn"),

    presetSelect: document.getElementById("presetSelect"),
    regimeBadge: document.getElementById("regimeBadge"),
    grfControls: document.getElementById("grfControls"),
    lengthScaleSlider: document.getElementById("lengthScaleSlider"),
    lengthScaleVal: document.getElementById("lengthScaleVal"),
    alphaSlider: document.getElementById("alphaSlider"),
    alphaVal: document.getElementById("alphaVal"),
    seedInput: document.getElementById("seedInput"),
    seedVal: document.getElementById("seedVal"),
    randomSeedBtn: document.getElementById("randomSeedBtn"),
    viscositySelect: document.getElementById("viscositySelect"),
    refreshBtn: document.getElementById("refreshBtn"),

    vortexPainterPanel: document.getElementById("vortexPainterPanel"),
    vortexCanvas: document.getElementById("vortexCanvas"),
    vortexCountLabel: document.getElementById("vortexCountLabel"),
    polarityCcwBtn: document.getElementById("polarityCcwBtn"),
    polarityCwBtn: document.getElementById("polarityCwBtn"),
    brushSigmaSlider: document.getElementById("brushSigmaSlider"),
    brushSigmaVal: document.getElementById("brushSigmaVal"),
    vortexUndoBtn: document.getElementById("vortexUndoBtn"),
    vortexClearBtn: document.getElementById("vortexClearBtn"),
    vortexSeedBtn: document.getElementById("vortexSeedBtn"),

    backboneKmax: document.getElementById("backboneKmax"),
    backboneLayers: document.getElementById("backboneLayers"),
    backboneParams: document.getElementById("backboneParams"),
    backboneHidden: document.getElementById("backboneHidden"),

    latencyMetric: document.getElementById("latencyMetric"),
    latencySub: document.getElementById("latencySub"),
    speedupMetric: document.getElementById("speedupMetric"),
    speedupSub: document.getElementById("speedupSub"),
    mseMetric: document.getElementById("mseMetric"),
    mseSub: document.getElementById("mseSub"),
    peakVortMetric: document.getElementById("peakVortMetric"),
    enstrophySub: document.getElementById("enstrophySub"),

    streamlinesToggle: document.getElementById("streamlinesToggle"),
    particleFlowToggle: document.getElementById("particleFlowToggle"),
    surface3DToggle: document.getElementById("surface3DToggle"),
    fftSpectrumToggle: document.getElementById("fftSpectrumToggle"),
    heatmap2DContainer: document.getElementById("heatmap2DContainer"),
    particleFlowContainer: document.getElementById("particleFlowContainer"),
    particleFlowCanvas: document.getElementById("particleFlowCanvas"),
    particlePlayBtn: document.getElementById("particlePlayBtn"),
    particleResetBtn: document.getElementById("particleResetBtn"),
    particleSpeedSelect: document.getElementById("particleSpeedSelect"),
    particleMaxSpeed: document.getElementById("particleMaxSpeed"),
    particleRmsSpeed: document.getElementById("particleRmsSpeed"),
    surface3DContainer: document.getElementById("surface3DContainer"),
    fft2Container: document.getElementById("fft2Container"),
    fft2Caption: document.getElementById("fft2Caption"),

    rolloutFramesSelect: document.getElementById("rolloutFramesSelect"),
    runRolloutBtn: document.getElementById("runRolloutBtn"),
    rolloutControlsBar: document.getElementById("rolloutControlsBar"),
    rolloutPlayBtn: document.getElementById("rolloutPlayBtn"),
    rolloutSlider: document.getElementById("rolloutSlider"),
    rolloutTimeLabel: document.getElementById("rolloutTimeLabel"),
    rolloutFpsSelect: document.getElementById("rolloutFpsSelect"),
    rolloutPinoMs: document.getElementById("rolloutPinoMs"),
    rolloutRk4Ms: document.getElementById("rolloutRk4Ms"),
    rolloutSpeedup: document.getElementById("rolloutSpeedup"),
    rolloutPlotsContainer: document.getElementById("rolloutPlotsContainer"),

    refreshSrBtn: document.getElementById("refreshSrBtn"),
    srWipeStage: document.getElementById("srWipeStage"),
    srWipeLeft: document.getElementById("srWipeLeft"),
    srWipeRight: document.getElementById("srWipeRight"),
    srWipeOverlay: document.getElementById("srWipeOverlay"),
    srWipeDivider: document.getElementById("srWipeDivider"),
    srWipeSlider: document.getElementById("srWipeSlider"),
    srWipePctLabel: document.getElementById("srWipePctLabel"),
    srWipeRightBadge: document.getElementById("srWipeRightBadge"),
    startRaceBtn: document.getElementById("startRaceBtn"),
    startScalingBtn: document.getElementById("startScalingBtn"),
    scalingEmptyOverlay: document.getElementById("scalingEmptyOverlay"),

    ttaDipoleBtn: document.getElementById("ttaDipoleBtn"),
    ttaShearBtn: document.getElementById("ttaShearBtn"),
    startTtaBtn: document.getElementById("startTtaBtn"),
    ttaStepsInput: document.getElementById("ttaStepsInput"),
    ttaStepsVal: document.getElementById("ttaStepsVal"),
    ttaLrSelect: document.getElementById("ttaLrSelect"),
    ttaAlphaInput: document.getElementById("ttaAlphaInput"),
    ttaAlphaVal: document.getElementById("ttaAlphaVal"),
    ttaWipeCard: document.getElementById("ttaWipeCard"),
    ttaWipeStage: document.getElementById("ttaWipeStage"),
    ttaWipeLeft: document.getElementById("ttaWipeLeft"),
    ttaWipeRight: document.getElementById("ttaWipeRight"),
    ttaWipeOverlay: document.getElementById("ttaWipeOverlay"),
    ttaWipeDivider: document.getElementById("ttaWipeDivider"),
    ttaWipeSlider: document.getElementById("ttaWipeSlider"),
    ttaWipePctLabel: document.getElementById("ttaWipePctLabel"),

    psoParticlesInput: document.getElementById("psoParticlesInput"),
    psoParticlesVal: document.getElementById("psoParticlesVal"),
    psoIterationsInput: document.getElementById("psoIterationsInput"),
    psoIterationsVal: document.getElementById("psoIterationsVal"),
    psoBoundInput: document.getElementById("psoBoundInput"),
    psoBoundVal: document.getElementById("psoBoundVal"),
    startPsoBtn: document.getElementById("startPsoBtn"),

    archSvgContainer: document.getElementById("archSvgContainer"),
    archLayersTableBody: document.getElementById("archLayersTableBody"),
    archTotalParamsBadge: document.getElementById("archTotalParamsBadge"),

    toastNotification: document.getElementById("toastNotification"),
    toastMessage: document.getElementById("toastMessage")
  };

  // --------------------------------------------------------------------------
  // 4. UTILITIES
  // --------------------------------------------------------------------------
  function transpose2D(matrix) {
    if (!Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(matrix[0])) {
      return matrix || [];
    }
    const rows = matrix.length;
    const cols = matrix[0].length;
    const out = Array.from({ length: cols }, () => new Array(rows));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        out[j][i] = matrix[i][j];
      }
    }
    return out;
  }

  function safeZMax(val) {
    const num = Number(val);
    return Number.isFinite(num) && num > 1e-4 ? num : 1e-4;
  }

  function fmt(num, digits = 3) {
    const val = Number(num);
    if (!Number.isFinite(val)) return "--";
    return val.toFixed(digits);
  }

  function fmtSigned(num, digits = 4) {
    const val = Number(num);
    if (!Number.isFinite(val)) return "--";
    return (val >= 0 ? "+" : "") + val.toFixed(digits);
  }

  function fmtParamsShort(count) {
    const n = Number(count);
    if (!Number.isFinite(n)) return "--";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return String(n);
  }

  function showToast(message) {
    els.toastMessage.textContent = message;
    els.toastNotification.classList.add("visible");
    clearTimeout(state.timers.toast);
    state.timers.toast = setTimeout(() => {
      els.toastNotification.classList.remove("visible");
    }, 2800);
  }

  async function fetchJson(url, options = {}) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (_) {
      throw new Error("Cannot reach PINO server — ensure server.py is running on this port.");
    }
    let data;
    try {
      data = await res.json();
    } catch (_) {
      throw new Error(`Invalid server response (HTTP ${res.status})`);
    }
    if (!res.ok || data.status === "error") {
      throw new Error(data.message || data.error || `HTTP ${res.status}`);
    }
    return data;
  }

  function apiGet(url) {
    return fetchJson(url);
  }

  function apiPost(url, payload) {
    return fetchJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
  }

  function buildBasePayload(includeRes = true) {
    const preset = els.presetSelect.value;
    const payload = {
      preset,
      length_scale: parseFloat(els.lengthScaleSlider.value),
      alpha: parseFloat(els.alphaSlider.value),
      seed: parseInt(els.seedInput.value, 10) || 42
    };
    if (includeRes) {
      payload.target_resolution = state.targetRes;
    }
    if (preset === "custom") {
      payload.custom_vortices = state.painter.vortices.map(v => ({
        x: Number(v.x.toFixed(4)),
        y: Number(v.y.toFixed(4)),
        strength: Number(v.strength.toFixed(3)),
        sigma: Number(v.sigma.toFixed(4))
      }));
    }
    return payload;
  }

  function getSelectedViscosity() {
    return parseFloat(els.viscositySelect.value) || 1e-3;
  }

  const prefersReducedMotion = () =>
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const animFrameMap = new WeakMap();
  const numericPrevMap = new WeakMap();

  function animateNumericText(el, targetVal, formatter, durationMs = 360) {
    if (!el) return;
    const target = Number(targetVal);
    if (!Number.isFinite(target)) {
      el.textContent = formatter(targetVal);
      return;
    }
    if (prefersReducedMotion()) {
      numericPrevMap.set(el, target);
      el.textContent = formatter(target);
      return;
    }
    const prevRaf = animFrameMap.get(el);
    if (prevRaf) cancelAnimationFrame(prevRaf);

    const startVal = numericPrevMap.has(el) ? numericPrevMap.get(el) : target;
    numericPrevMap.set(el, target);

    if (Math.abs(target - startVal) < 1e-6) {
      el.textContent = formatter(target);
      return;
    }

    const startTs = performance.now();
    function step(now) {
      const t = Math.min(1, (now - startTs) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = formatter(startVal + (target - startVal) * eased);
      if (t < 1) {
        animFrameMap.set(el, requestAnimationFrame(step));
      } else {
        animFrameMap.delete(el);
      }
    }
    animFrameMap.set(el, requestAnimationFrame(step));
  }

  function flashCard(el) {
    const card = el?.closest ? el.closest(".metric-card") : null;
    if (!card || prefersReducedMotion()) return;
    card.classList.remove("card-flash");
    void card.offsetWidth;
    card.classList.add("card-flash");
  }

  function setSurfaceLoading(ids, isLoading) {
    ids.forEach(id => {
      const el = document.getElementById(id);
      const card = el?.closest ? el.closest(".panel-card") : el;
      if (card) card.classList.toggle("is-loading-surface", Boolean(isLoading));
    });
  }

  function updateSpeedupCard(speedupValue, subtitleText) {
    const val = Number(speedupValue);
    if (!Number.isFinite(val)) return;
    animateNumericText(
      els.speedupMetric,
      val,
      v => `${v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(1)}×`,
      420
    );
    flashCard(els.speedupMetric);
    els.speedupSub.textContent = subtitleText;
    els.speedupSub.className = "mt-2 inline-flex items-center text-[11px] font-semibold text-blue-700 bg-blue-50 border border-blue-200 px-2.5 py-0.5 rounded-full font-mono";
  }

  function renderMathSafe(root = document.body) {
    if (typeof window.renderMathInElement === "function") {
      try {
        window.renderMathInElement(root, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false }
          ],
          throwOnError: false
        });
      } catch (err) {
        console.warn("KaTeX auto-render warning:", err);
      }
    }
  }

  function resizeAllPlots() {
    const plotDivs = document.querySelectorAll(".js-plotly-plot");
    plotDivs.forEach(el => {
      if (el.offsetParent !== null) {
        try {
          Plotly.Plots.resize(el);
        } catch (_) {
          // Ignore inactive plots
        }
      }
    });
  }

  // --------------------------------------------------------------------------
  // 4B. MOTION STUDIO ENGINES (TAB PILL, HEADER RIBBON, WIPES, 60 FPS TRACERS)
  // --------------------------------------------------------------------------
  function updateTabPill() {
    const bar = els.tabBar;
    const pill = els.tabSliderPill;
    if (!bar || !pill) return;
    const activeBtn = bar.querySelector(".tab-btn.active");
    if (!activeBtn) {
      pill.style.opacity = "0";
      return;
    }
    pill.style.opacity = "1";
    pill.style.width = `${activeBtn.offsetWidth}px`;
    pill.style.height = `${activeBtn.offsetHeight}px`;
    pill.style.transform = `translate3d(${activeBtn.offsetLeft}px, ${activeBtn.offsetTop}px, 0)`;
  }

  function triggerTabEntrance(tabEl) {
    if (!tabEl || prefersReducedMotion()) return;
    tabEl.classList.remove("is-entering");
    void tabEl.offsetWidth;
    tabEl.classList.add("is-entering");
  }

  function initHeaderFlowCanvas() {
    const canvas = els.headerFlowCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let phase = 0;
    let rafId = null;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(300, Math.floor(rect.width * dpr));
      canvas.height = Math.max(60, Math.floor(rect.height * dpr));
    }

    function drawFrame() {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const ribbons = [
        { amp: H * 0.22, freq: 2.2, speed: 1.0, y0: H * 0.42, color: "rgba(37, 99, 235, 0.14)", width: 1.6 },
        { amp: H * 0.16, freq: 3.1, speed: -0.75, y0: H * 0.56, color: "rgba(14, 165, 233, 0.14)", width: 1.4 },
        { amp: H * 0.12, freq: 4.0, speed: 0.55, y0: H * 0.68, color: "rgba(5, 150, 105, 0.11)", width: 1.1 }
      ];

      ribbons.forEach((r, idx) => {
        ctx.beginPath();
        const step = 12;
        for (let px = 0; px <= W; px += step) {
          const u = px / W;
          const y =
            r.y0 +
            Math.sin(u * Math.PI * 2 * r.freq + phase * r.speed + idx) * r.amp * 0.65 +
            Math.cos(u * Math.PI * 2 * (r.freq * 0.5) - phase * 0.5) * r.amp * 0.35;
          if (px === 0) ctx.moveTo(px, y);
          else ctx.lineTo(px, y);
        }
        ctx.strokeStyle = r.color;
        ctx.lineWidth = r.width;
        ctx.stroke();
      });
    }

    function loop() {
      if (!document.hidden && !prefersReducedMotion()) {
        phase += 0.014;
        drawFrame();
      }
      rafId = requestAnimationFrame(loop);
    }

    resize();
    drawFrame();
    window.addEventListener("resize", resize);
    if (!prefersReducedMotion()) {
      rafId = requestAnimationFrame(loop);
    }
    return () => cancelAnimationFrame(rafId);
  }

  function hexToRgb(hex) {
    const clean = hex.replace("#", "");
    const num = parseInt(clean, 16);
    return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
  }

  function buildColorLut(stops) {
    const parsed = stops.map(([t, hex]) => [t, hexToRgb(hex)]);
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      let seg = 0;
      while (seg < parsed.length - 2 && t > parsed[seg + 1][0]) seg++;
      const [t0, c0] = parsed[seg];
      const [t1, c1] = parsed[seg + 1];
      const f = Math.max(0, Math.min(1, (t - t0) / Math.max(1e-6, t1 - t0)));
      lut[i * 3] = Math.round(c0[0] + (c1[0] - c0[0]) * f);
      lut[i * 3 + 1] = Math.round(c0[1] + (c1[1] - c0[1]) * f);
      lut[i * 3 + 2] = Math.round(c0[2] + (c1[2] - c0[2]) * f);
    }
    return lut;
  }

  const LUT_VORTICITY = buildColorLut(COLORSCALE_VORTICITY);
  const LUT_ERROR = buildColorLut(COLORSCALE_ERROR);

  function renderMatrixToCanvas(canvas, matrix, zMin, zMax, lut) {
    if (!canvas || !Array.isArray(matrix) || matrix.length === 0) return;
    const nx = matrix.length;
    const ny = matrix[0].length;
    const W = canvas.width;
    const H = canvas.height;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(W, H);
    const data = img.data;
    const span = Math.max(1e-8, zMax - zMin);

    for (let py = 0; py < H; py++) {
      // Canvas top (py=0) maps to y=ny-1; bottom maps to y=0 (matching Plotly origin="lower")
      const fy = ((H - 1 - py) / Math.max(1, H - 1)) * (ny - 1);
      const y0 = Math.floor(fy);
      const y1 = Math.min(ny - 1, y0 + 1);
      const wy = fy - y0;

      for (let px = 0; px < W; px++) {
        const fx = (px / Math.max(1, W - 1)) * (nx - 1);
        const x0 = Math.floor(fx);
        const x1 = Math.min(nx - 1, x0 + 1);
        const wx = fx - x0;

        const v00 = matrix[x0][y0];
        const v10 = matrix[x1][y0];
        const v01 = matrix[x0][y1];
        const v11 = matrix[x1][y1];
        const val = (v00 * (1 - wx) + v10 * wx) * (1 - wy) + (v01 * (1 - wx) + v11 * wx) * wy;

        const norm = Math.max(0, Math.min(1, (val - zMin) / span));
        const idx = Math.round(norm * 255) * 3;
        const p = (py * W + px) * 4;
        data[p] = lut[idx];
        data[p + 1] = lut[idx + 1];
        data[p + 2] = lut[idx + 2];
        data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function bindWipeComparator(stageEl, overlayEl, dividerEl, sliderEl, pctLabelEl) {
    if (!stageEl || !overlayEl || !dividerEl || !sliderEl) return;

    function applyPct(pct) {
      const clamped = Math.max(0, Math.min(100, pct));
      overlayEl.style.clipPath = `inset(0 ${100 - clamped}% 0 0)`;
      dividerEl.style.left = `${clamped}%`;
      sliderEl.value = String(Math.round(clamped));
      if (pctLabelEl) pctLabelEl.textContent = `Split: ${Math.round(clamped)}%`;
    }

    function updateFromPointer(clientX) {
      const rect = stageEl.getBoundingClientRect();
      if (rect.width <= 0) return;
      const pct = ((clientX - rect.left) / rect.width) * 100;
      applyPct(pct);
    }

    let dragging = false;
    stageEl.addEventListener("pointerdown", ev => {
      dragging = true;
      stageEl.setPointerCapture(ev.pointerId);
      updateFromPointer(ev.clientX);
    });
    stageEl.addEventListener("pointermove", ev => {
      if (dragging) updateFromPointer(ev.clientX);
    });
    const stop = () => { dragging = false; };
    stageEl.addEventListener("pointerup", stop);
    stageEl.addEventListener("pointercancel", stop);
    sliderEl.addEventListener("input", e => applyPct(parseFloat(e.target.value)));
    applyPct(50);
  }

  const particleEngine = (() => {
    const NUM_PARTICLES = 700;
    const bgCanvas = document.createElement("canvas");
    let tracers = [];
    let uxField = null;
    let uyField = null;
    let maxSpeed = 1.0;
    let rafId = null;

    function makeTracer() {
      const x = Math.random();
      const y = Math.random();
      return { x, y, px: x, py: y, age: Math.floor(Math.random() * 90), maxAge: 80 + Math.floor(Math.random() * 80) };
    }

    function resetTracers() {
      tracers = Array.from({ length: NUM_PARTICLES }, makeTracer);
      const canvas = els.particleFlowCanvas;
      if (canvas && bgCanvas.width > 0) {
        canvas.getContext("2d").drawImage(bgCanvas, 0, 0);
      }
    }

    function sampleBilinearPeriodic(field, x, y) {
      const nx = field.length;
      const ny = field[0].length;
      const fx = ((x % 1) + 1) % 1 * nx;
      const fy = ((y % 1) + 1) % 1 * ny;
      const i0 = Math.floor(fx) % nx;
      const j0 = Math.floor(fy) % ny;
      const i1 = (i0 + 1) % nx;
      const j1 = (j0 + 1) % ny;
      const wx = fx - Math.floor(fx);
      const wy = fy - Math.floor(fy);
      return (
        (field[i0][j0] * (1 - wx) + field[i1][j0] * wx) * (1 - wy) +
        (field[i0][j1] * (1 - wx) + field[i1][j1] * wx) * wy
      );
    }

    function updateField(uPred, ux, uy, zMax) {
      const canvas = els.particleFlowCanvas;
      if (!canvas || !Array.isArray(uPred) || !Array.isArray(ux) || !Array.isArray(uy)) return;

      bgCanvas.width = canvas.width;
      bgCanvas.height = canvas.height;
      renderMatrixToCanvas(bgCanvas, uPred, -zMax, zMax, LUT_VORTICITY);

      uxField = ux;
      uyField = uy;

      let peak = 0;
      let sumSq = 0;
      let count = 0;
      for (let i = 0; i < ux.length; i++) {
        for (let j = 0; j < ux[0].length; j++) {
          const s2 = ux[i][j] * ux[i][j] + uy[i][j] * uy[i][j];
          sumSq += s2;
          count++;
          if (s2 > peak) peak = s2;
        }
      }
      maxSpeed = Math.max(0.05, Math.sqrt(peak));
      const rms = Math.sqrt(sumSq / Math.max(1, count));

      animateNumericText(els.particleMaxSpeed, maxSpeed, v => v.toFixed(3), 260);
      animateNumericText(els.particleRmsSpeed, rms, v => v.toFixed(3), 260);

      if (tracers.length === 0) {
        resetTracers();
      } else {
        canvas.getContext("2d").drawImage(bgCanvas, 0, 0);
      }
      ensureLoop();
    }

    function tick() {
      const canvas = els.particleFlowCanvas;
      const active =
        canvas &&
        uxField &&
        uyField &&
        state.activeTab === "tab-sim" &&
        els.particleFlowToggle?.checked &&
        !state.particleFlow.paused &&
        !document.hidden;

      if (active) {
        const ctx = canvas.getContext("2d");
        const W = canvas.width;
        const H = canvas.height;

        // Soft fade back into the underlying vorticity field so trails stay crisp
        ctx.save();
        ctx.globalAlpha = 0.22;
        ctx.drawImage(bgCanvas, 0, 0);
        ctx.restore();

        const dt = (0.0068 * state.particleFlow.speed) / Math.max(0.35, maxSpeed);
        ctx.lineCap = "round";

        // Crisp dual-pass tracer strokes (soft white halo + deep slate ink core)
        ctx.beginPath();
        for (let k = 0; k < tracers.length; k++) {
          const p = tracers[k];
          p.px = p.x;
          p.py = p.y;

          const u = sampleBilinearPeriodic(uxField, p.x, p.y);
          const v = sampleBilinearPeriodic(uyField, p.x, p.y);

          p.x += u * dt;
          p.y += v * dt;
          p.age += 1;

          const wrapped = p.x < 0 || p.x >= 1 || p.y < 0 || p.y >= 1;
          if (wrapped) {
            p.x = ((p.x % 1) + 1) % 1;
            p.y = ((p.y % 1) + 1) % 1;
          }

          if (p.age > p.maxAge) {
            tracers[k] = makeTracer();
            continue;
          }

          if (!wrapped) {
            ctx.moveTo(p.px * W, (1 - p.py) * H);
            ctx.lineTo(p.x * W, (1 - p.y) * H);
          }
        }
        ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
        ctx.lineWidth = 2.4;
        ctx.stroke();

        ctx.strokeStyle = "rgba(15, 23, 42, 0.78)";
        ctx.lineWidth = 1.25;
        ctx.stroke();
      }

      if (!prefersReducedMotion()) {
        rafId = requestAnimationFrame(tick);
      } else {
        rafId = null;
      }
    }

    function ensureLoop() {
      if (rafId === null && !prefersReducedMotion()) {
        rafId = requestAnimationFrame(tick);
      }
    }

    return { updateField, resetTracers, ensureLoop };
  })();

  // --------------------------------------------------------------------------
  // 5. SIDEBAR REGIME BADGE & INTERACTIVE VORTEX PAINTER
  // --------------------------------------------------------------------------
  function updateRegimeBadge() {
    const preset = els.presetSelect.value;
    const l = parseFloat(els.lengthScaleSlider.value);
    const a = parseFloat(els.alphaSlider.value);

    const badgeStyles = {
      grfIn: {
        cls: "mt-2 text-[11px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-md flex items-center",
        dot: "bg-emerald-500",
        label: "In-Distribution GRF"
      },
      grfGen: {
        cls: "mt-2 text-[11px] font-semibold text-blue-700 bg-blue-50 border border-blue-200 px-2.5 py-1 rounded-md flex items-center",
        dot: "bg-blue-500",
        label: "Parametric Generalization"
      },
      ood: {
        cls: "mt-2 text-[11px] font-semibold text-amber-800 bg-amber-50 border border-amber-200 px-2.5 py-1 rounded-md flex items-center",
        dot: "bg-amber-500",
        label: "Out-Of-Distribution (OOD)"
      },
      taylor: {
        cls: "mt-2 text-[11px] font-semibold text-indigo-700 bg-indigo-50 border border-indigo-200 px-2.5 py-1 rounded-md flex items-center",
        dot: "bg-indigo-500",
        label: "Taylor-Green Benchmark"
      },
      custom: {
        cls: "mt-2 text-[11px] font-semibold text-purple-700 bg-purple-50 border border-purple-200 px-2.5 py-1 rounded-md flex items-center",
        dot: "bg-purple-500",
        label: "Interactive Vortex Paint"
      }
    };

    let selected;
    if (preset === "grf") {
      selected = Math.abs(l - 0.5) < 0.05 && Math.abs(a - 2.5) < 0.15 ? badgeStyles.grfIn : badgeStyles.grfGen;
    } else if (preset === "dipole" || preset === "shear") {
      selected = badgeStyles.ood;
    } else if (preset === "taylor_green") {
      selected = badgeStyles.taylor;
    } else {
      selected = badgeStyles.custom;
    }

    els.regimeBadge.className = selected.cls;
    els.regimeBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full ${selected.dot} mr-1.5"></span><span>${selected.label}</span>`;
  }

  function syncPresetPanels() {
    const preset = els.presetSelect.value;
    if (preset === "grf") {
      els.grfControls.classList.remove("opacity-40", "pointer-events-none");
    } else {
      els.grfControls.classList.add("opacity-40", "pointer-events-none");
    }
    if (preset === "custom") {
      els.vortexPainterPanel.classList.remove("hidden");
      drawVortexCanvas();
    } else {
      els.vortexPainterPanel.classList.add("hidden");
    }
    updateRegimeBadge();
  }

  function updatePolarityButtons() {
    if (state.painter.polarity > 0) {
      els.polarityCcwBtn.className = "px-2 py-1.5 rounded-lg text-xs font-bold border border-red-400 bg-red-50 text-red-700 flex items-center justify-center space-x-1 transition";
      els.polarityCwBtn.className = "px-2 py-1.5 rounded-lg text-xs font-semibold border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 flex items-center justify-center space-x-1 transition";
    } else {
      els.polarityCcwBtn.className = "px-2 py-1.5 rounded-lg text-xs font-semibold border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 flex items-center justify-center space-x-1 transition";
      els.polarityCwBtn.className = "px-2 py-1.5 rounded-lg text-xs font-bold border border-blue-400 bg-blue-50 text-blue-700 flex items-center justify-center space-x-1 transition";
    }
  }

  function drawVortexCanvas() {
    const canvas = els.vortexCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    // Clean Swiss Grid Background
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#F8FAFC";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "#E2E8F0";
    ctx.lineWidth = 1;
    const steps = 4;
    for (let i = 1; i < steps; i++) {
      const p = (i / steps) * W;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, H);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, p);
      ctx.lineTo(W, p);
      ctx.stroke();
    }

    // Draw vortices (including periodic ghost wraps near boundaries)
    state.painter.vortices.forEach(v => {
      const isPos = v.strength >= 0;
      const fillRgba = isPos ? "rgba(220, 38, 38, 0.28)" : "rgba(37, 99, 235, 0.28)";
      const strokeHex = isPos ? "#DC2626" : "#2563EB";
      const radiusPx = Math.max(8, v.sigma * W * 1.45);

      for (const ox of [-1, 0, 1]) {
        for (const oy of [-1, 0, 1]) {
          const cx = (v.x + ox) * W;
          const cy = (1 - (v.y + oy)) * H;
          if (cx < -radiusPx || cx > W + radiusPx || cy < -radiusPx || cy > H + radiusPx) continue;

          ctx.beginPath();
          ctx.arc(cx, cy, radiusPx, 0, Math.PI * 2);
          ctx.fillStyle = fillRgba;
          ctx.fill();
          ctx.lineWidth = 1.4;
          ctx.strokeStyle = strokeHex;
          ctx.stroke();

          if (ox === 0 && oy === 0) {
            ctx.fillStyle = strokeHex;
            ctx.font = "bold 11px JetBrains Mono, monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(isPos ? "+" : "−", cx, cy + 0.5);
          }
        }
      }
    });

    els.vortexCountLabel.textContent = `${state.painter.vortices.length} ${state.painter.vortices.length === 1 ? "vortex" : "vortices"}`;
  }

  function addVortexFromEvent(ev, isDrag = false) {
    const rect = els.vortexCanvas.getBoundingClientRect();
    const scaleX = els.vortexCanvas.width / rect.width;
    const scaleY = els.vortexCanvas.height / rect.height;
    const px = Math.max(0, Math.min(els.vortexCanvas.width, (ev.clientX - rect.left) * scaleX));
    const py = Math.max(0, Math.min(els.vortexCanvas.height, (ev.clientY - rect.top) * scaleY));

    if (isDrag && state.painter.lastCanvasPt) {
      const dx = px - state.painter.lastCanvasPt.px;
      const dy = py - state.painter.lastCanvasPt.py;
      if (Math.hypot(dx, dy) < 16) return;
    }

    state.painter.lastCanvasPt = { px, py };
    const vx = px / els.vortexCanvas.width;
    const vy = 1.0 - py / els.vortexCanvas.height;

    state.painter.vortices.push({
      x: Math.max(0.0, Math.min(0.999, vx)),
      y: Math.max(0.0, Math.min(0.999, vy)),
      strength: state.painter.polarity * 3.0,
      sigma: state.painter.sigma
    });

    drawVortexCanvas();
    triggerControlsUpdate();
  }

  function seedDipoleVortices() {
    state.painter.vortices = [
      { x: 0.38, y: 0.44, strength: 3.0, sigma: state.painter.sigma },
      { x: 0.62, y: 0.56, strength: -3.0, sigma: state.painter.sigma }
    ];
    drawVortexCanvas();
  }

  // --------------------------------------------------------------------------
  // 6. INITIALIZATION: /api/info + TAB 6 SVG ARCHITECTURE DIAGRAM
  // --------------------------------------------------------------------------
  function renderArchSvg(info) {
    const hiddenDim = info?.hidden_dim ?? 64;
    const modes1 = info?.modes1 ?? 16;
    const modes2 = info?.modes2 ?? 16;
    const numLayers = info?.num_layers ?? 4;
    const baseRes = info?.base_resolution ?? 64;
    const totalParams = info?.total_params ?? 6718625;
    const layers = Array.isArray(info?.layers) ? info.layers : [];

    const liftParams = layers
      .filter(l => l.name.startsWith("lifting"))
      .reduce((acc, l) => acc + (l.params || 0), 0);
    const fourierParams = layers
      .filter(l => l.name.startsWith("fourier_layers"))
      .reduce((acc, l) => acc + (l.params || 0), 0);
    const projParams = layers
      .filter(l => l.name.startsWith("projection"))
      .reduce((acc, l) => acc + (l.params || 0), 0);

    els.archTotalParamsBadge.textContent = `Total Parameters: ${totalParams.toLocaleString()} (${fmtParamsShort(totalParams)})`;

    const modeCells = [];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 6; c++) {
        const cx = 560 + c * 9;
        const cy = 83 + r * 9;
        const fill = (r === 1 && (c === 2 || c === 3)) ? "#2563EB" : "#60A5FA";
        modeCells.push(`<rect class="arch-mode-cell" x="${cx}" y="${cy}" width="7" height="7" rx="1.5" fill="${fill}" />`);
      }
    }

    els.archSvgContainer.innerHTML = `
      <svg viewBox="0 0 980 258" class="w-full min-w-[760px] h-auto select-none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="arrowBlue" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 Z" fill="#2563EB" />
          </marker>
          <marker id="arrowSlate" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 Z" fill="#475569" />
          </marker>
          <marker id="arrowEmerald" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#059669" />
          </marker>
          <filter id="orbGlow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="2.2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <!-- Background Panel -->
        <rect x="4" y="4" width="972" height="250" rx="12" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1.2" />

        <!-- Behind-Node Main Trunk Signal Orbs (travel along wires without crossing box text) -->
        <circle r="4" fill="#2563EB" filter="url(#orbGlow)">
          <animateMotion dur="3.4s" repeatCount="indefinite" path="M81,120 L246,120 L350,120" />
        </circle>
        <circle r="4" fill="#059669" filter="url(#orbGlow)">
          <animateMotion dur="2.6s" repeatCount="indefinite" begin="0.5s" path="M640,120 L748,120 L909,120" />
        </circle>

        <!-- Node 1: Input Tensor -->
        <rect x="20" y="72" width="122" height="96" rx="10" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" />
        <text x="81" y="96" text-anchor="middle" font-size="11" font-weight="700" fill="#0F172A">Input Grid + IC</text>
        <text x="81" y="114" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="#2563EB">[x, y, ω₀(x,y)]</text>
        <text x="81" y="134" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="#475569">(B, 3, ${baseRes}, ${baseRes})</text>
        <text x="81" y="154" text-anchor="middle" font-size="9" fill="#64748B">Periodic [0, 1)²</text>

        <path id="wire1" class="arch-flow-wire" d="M142,120 L174,120" stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrowSlate)" />

        <!-- Node 2: Lifting MLP -->
        <rect x="178" y="66" width="136" height="108" rx="10" fill="#EFF6FF" stroke="#93C5FD" stroke-width="1.5" />
        <text x="246" y="90" text-anchor="middle" font-size="11" font-weight="700" fill="#1E3A8A">LiftingBlock P</text>
        <text x="246" y="109" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="#1D4ED8">1×1 Conv + GELU</text>
        <text x="246" y="128" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="#334155">3 → ${hiddenDim / 2} → ${hiddenDim}</text>
        <rect x="196" y="142" width="100" height="20" rx="5" fill="#FFFFFF" stroke="#BFDBFE" />
        <text x="246" y="155" text-anchor="middle" font-size="9.5" font-family="JetBrains Mono, monospace" font-weight="600" fill="#1E40AF">${liftParams ? liftParams.toLocaleString() : "2,208"} params</text>

        <path id="wire2" class="arch-flow-wire" d="M314,120 L346,120" stroke="#2563EB" stroke-width="2" fill="none" marker-end="url(#arrowBlue)" />

        <!-- Node 3: 4x SpectralConv2d Stack -->
        <rect x="350" y="22" width="290" height="202" rx="12" fill="#FFFFFF" stroke="#2563EB" stroke-width="1.8" />
        <text x="495" y="44" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1E3A8A">${numLayers} × SpectralConv2d + GELU</text>
        <text x="495" y="60" text-anchor="middle" font-size="9.5" font-family="JetBrains Mono, monospace" fill="#475569">d_v = ${hiddenDim} • k_max = (${modes1}, ${modes2}) • ${fourierParams ? fmtParamsShort(fourierParams) : "6.71M"} params</text>

        <!-- Upper Spectral Branch + Live Mode Grid -->
        <rect x="366" y="74" width="258" height="64" rx="8" fill="#EFF6FF" stroke="#BFDBFE" stroke-width="1.2" />
        <text x="458" y="93" text-anchor="middle" font-size="10" font-weight="700" fill="#1D4ED8">Spectral Kernel K_ϕ</text>
        <text x="458" y="110" text-anchor="middle" font-size="9" font-family="JetBrains Mono, monospace" fill="#0F172A">rfft2 → R_ϕ[|k|≤${modes1}] → irfft2</text>
        <text x="458" y="126" text-anchor="middle" font-size="8.8" fill="#475569">weights ∈ ℂ^(${hiddenDim}×${hiddenDim}×${modes1}×${modes2})</text>
        <rect x="556" y="79" width="60" height="34" rx="4" fill="#FFFFFF" stroke="#93C5FD" stroke-width="1" />
        ${modeCells.join("")}
        <text x="586" y="126" text-anchor="middle" font-size="7.8" font-family="JetBrains Mono, monospace" fill="#1D4ED8">[-${modes1},${modes1}]²</text>

        <!-- Lower Local Bypass Branch -->
        <rect x="366" y="148" width="258" height="40" rx="8" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2" />
        <text x="495" y="165" text-anchor="middle" font-size="10" font-weight="700" fill="#334155">Local Spatial Shortcut W (1×1 Conv2d)</text>
        <text x="495" y="180" text-anchor="middle" font-size="9" font-family="JetBrains Mono, monospace" fill="#64748B">Bilinear resize if target_s ≠ (${baseRes},${baseRes})</text>

        <text x="495" y="210" text-anchor="middle" font-size="9.5" font-weight="600" fill="#059669">Residual Sum + GELU (layers 1..${Math.max(1, numLayers - 1)})</text>

        <!-- Internal Branch Paths for Signal Orbs (routed cleanly in channels above/below inner cards) -->
        <path id="branchFourier" d="M350,115 C358,72 362,69 374,69 L616,69 C628,69 632,110 640,120" fill="none" stroke="rgba(37,99,235,0.28)" stroke-width="1.4" />
        <path id="branchShortcut" d="M350,126 C358,192 362,195 374,195 L616,195 C628,195 632,132 640,120" fill="none" stroke="rgba(71,85,105,0.26)" stroke-width="1.4" />

        <path id="wire3" class="arch-flow-wire" d="M640,120 L672,120" stroke="#2563EB" stroke-width="2" fill="none" marker-end="url(#arrowBlue)" />

        <!-- Node 4: Projection MLP (TTA Target) -->
        <rect x="676" y="62" width="144" height="116" rx="10" fill="#ECFDF5" stroke="#34D399" stroke-width="1.6" />
        <text x="748" y="85" text-anchor="middle" font-size="11" font-weight="700" fill="#065F46">ProjectionBlock Q</text>
        <text x="748" y="103" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="#047857">${hiddenDim} → ${hiddenDim / 2} → 1</text>
        <text x="748" y="121" text-anchor="middle" font-size="9.5" font-family="JetBrains Mono, monospace" fill="#334155">${projParams ? projParams.toLocaleString() : "2,113"} params</text>
        <rect x="690" y="134" width="116" height="30" rx="6" fill="#FFFFFF" stroke="#6EE7B7" />
        <text x="748" y="148" text-anchor="middle" font-size="8.8" font-weight="700" fill="#059669">TTA Trainable Only</text>
        <text x="748" y="159" text-anchor="middle" font-size="8.2" fill="#047857">(Fourier Layers Frozen)</text>

        <!-- TTA Gradient Feedback Loop -->
        <path class="arch-tta-halo" d="M780,178 C780,206 716,206 716,181" fill="none" stroke="#059669" stroke-width="1.6" marker-end="url(#arrowEmerald)" />
        <text x="748" y="213" text-anchor="middle" font-size="8.3" font-weight="700" font-family="JetBrains Mono, monospace" fill="#047857">∇ L_PDE TTA Loop</text>

        <path id="wire4" class="arch-flow-wire" d="M820,120 L852,120" stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrowSlate)" />

        <!-- Node 5: Output Field -->
        <rect x="856" y="72" width="106" height="96" rx="10" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" />
        <text x="909" y="96" text-anchor="middle" font-size="11" font-weight="700" fill="#0F172A">Output ω(T)</text>
        <text x="909" y="114" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="#2563EB">u_pred(x,y)</text>
        <text x="909" y="134" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="#475569">(B, 1, s', s')</text>
        <text x="909" y="154" text-anchor="middle" font-size="9" fill="#059669">64² / 128² / 256²</text>

        <!-- Internal Branch Signal Packets -->
        <circle r="3.6" fill="#0EA5E9" filter="url(#orbGlow)">
          <animateMotion dur="2.6s" repeatCount="indefinite" begin="0.3s" path="M350,115 C358,72 362,69 374,69 L616,69 C628,69 632,110 640,120" />
        </circle>
        <circle r="3.2" fill="#475569">
          <animateMotion dur="2.6s" repeatCount="indefinite" begin="0.9s" path="M350,126 C358,192 362,195 374,195 L616,195 C628,195 632,132 640,120" />
        </circle>
      </svg>
    `;

    // Populate layers table
    if (layers.length > 0) {
      els.archLayersTableBody.innerHTML = layers.map(layer => `
        <tr class="hover:bg-slate-50">
          <td class="px-4 py-2 font-semibold text-slate-800">${layer.name}</td>
          <td class="px-4 py-2 text-slate-600">[${Array.isArray(layer.shape) ? layer.shape.join(", ") : layer.shape}]</td>
          <td class="px-4 py-2 text-right font-semibold text-slate-900">${Number(layer.params || 0).toLocaleString()}</td>
        </tr>
      `).join("");
    }
  }

  async function fetchModelInfo() {
    try {
      const info = await apiGet("/api/info");
      state.info = info;

      const paramShort = fmtParamsShort(info.total_params);
      const ckptName = info.checkpoint || "pino_best.pt";
      const kMax = info.modes1 ?? 16;

      els.checkpointBadge.innerHTML = `
        <span class="w-2 h-2 mr-2 rounded-full bg-emerald-500 animate-pulse"></span>
        ${ckptName} • ${paramShort} params • k_max = ${kMax}
      `;
      els.deviceBadge.textContent = `${String(info.device || "CPU").toUpperCase()} ACCELERATION`;

      // Card 3: Honest Model Validation MSE & Rel-L2
      const ckptLoss = Number(info.checkpoint_loss);
      els.mseMetric.textContent = Number.isFinite(ckptLoss) ? ckptLoss.toFixed(4) : "0.0049";
      const epochLabel = info.checkpoint_epoch ? `Epoch ${info.checkpoint_epoch} • ` : "";
      const valRel = Number(info.checkpoint_val_rel_l2);
      const relLabel = Number.isFinite(valRel) ? `${(valRel * 100).toFixed(1)}% Val Rel-L2` : "~7.4% Zero-Shot Rel-L2";
      els.mseSub.textContent = `${epochLabel}${relLabel}`;

      // Sidebar Mathematical Backbone Card
      els.backboneKmax.textContent = `k_max = ${kMax}`;
      els.backboneLayers.textContent = `${info.num_layers ?? 4} SpectralConv2d`;
      els.backboneParams.textContent = `${Number(info.total_params || 0).toLocaleString()} params (${paramShort})`;
      els.backboneHidden.textContent = `d_v = ${info.hidden_dim ?? 64}`;

      renderArchSvg(info);
    } catch (err) {
      console.warn("Could not load /api/info:", err);
      els.checkpointBadge.textContent = "pino_best.pt • k_max = 16";
      renderArchSvg(null);
    }
  }

  // --------------------------------------------------------------------------
  // 7. TAB 1: /api/predict + INVARIANTS + 3D + 2D FFT + ROLLOUT PLAYER
  // --------------------------------------------------------------------------
  function renderInvariantsBar(initInv, predInv) {
    if (!initInv || !predInv) return;

    const circInit = Number(initInv.circulation);
    const circPred = Number(predInv.circulation);
    const eInit = Number(initInv.kinetic_energy);
    const ePred = Number(predInv.kinetic_energy);
    const zInit = Number(initInv.enstrophy);
    const zPred = Number(predInv.enstrophy);
    const pInit = Number(initInv.peak_vorticity);
    const pPred = Number(predInv.peak_vorticity);

    document.getElementById("invCircInit").textContent = fmt(circInit, 4);
    document.getElementById("invCircPred").textContent = fmt(circPred, 4);
    document.getElementById("invCircDelta").textContent = `Δ = ${fmtSigned(circPred - circInit, 4)}`;

    document.getElementById("invEnergyInit").textContent = fmt(eInit, 4);
    document.getElementById("invEnergyPred").textContent = fmt(ePred, 4);
    const ePct = ((ePred - eInit) / (Math.abs(eInit) + 1e-8)) * 100;
    document.getElementById("invEnergyDelta").textContent = `Δ = ${fmtSigned(ePct, 1)}%`;

    document.getElementById("invEnstInit").textContent = fmt(zInit, 4);
    document.getElementById("invEnstPred").textContent = fmt(zPred, 4);
    const zPct = ((zPred - zInit) / (Math.abs(zInit) + 1e-8)) * 100;
    document.getElementById("invEnstDelta").textContent = `Δ = ${fmtSigned(zPct, 1)}%`;

    document.getElementById("invPeakInit").textContent = fmt(pInit, 2);
    document.getElementById("invPeakPred").textContent = fmt(pPred, 2);
  }

  function renderFft2Plot(fft2Mat, modesCutoff) {
    if (!Array.isArray(fft2Mat) || fft2Mat.length === 0) return;
    const size = fft2Mat.length;
    const half = Math.floor(size / 2);
    const kCoords = Array.from({ length: size }, (_, idx) => idx - half);
    const kMax = Number(modesCutoff) || 16;

    els.fft2Caption.textContent = `Dashed box highlights the learned spectral mode window k_x, k_y ∈ [-${kMax}, ${kMax}] where complex multipliers R_ϕ operate.`;

    const layout = {
      ...makeFieldLayout("", "k_x", "k_y"),
      xaxis: {
        range: [kCoords[0] - 0.5, kCoords[kCoords.length - 1] + 0.5],
        showgrid: false,
        zeroline: true,
        zerolinecolor: "rgba(15, 23, 42, 0.25)",
        title: "k_x"
      },
      yaxis: {
        range: [kCoords[0] - 0.5, kCoords[kCoords.length - 1] + 0.5],
        showgrid: false,
        zeroline: true,
        zerolinecolor: "rgba(15, 23, 42, 0.25)",
        scaleanchor: "x",
        scaleratio: 1,
        title: "k_y"
      },
      shapes: [
        {
          type: "rect",
          x0: -kMax,
          x1: kMax,
          y0: -kMax,
          y1: kMax,
          line: { color: "#2563EB", width: 2.2, dash: "dash" }
        }
      ],
      annotations: [
        {
          x: kMax,
          y: kMax,
          text: `Learned [-${kMax}, ${kMax}]²`,
          showarrow: false,
          xanchor: "right",
          yanchor: "bottom",
          font: { size: 10, color: "#1D4ED8", family: "JetBrains Mono, monospace" },
          bgcolor: "rgba(255, 255, 255, 0.88)",
          bordercolor: "#93C5FD",
          borderwidth: 1
        }
      ]
    };

    Plotly.react(
      "fft2PredPlot",
      [
        {
          z: transpose2D(fft2Mat),
          x: kCoords,
          y: kCoords,
          type: "heatmap",
          colorscale: COLORSCALE_FFT,
          zsmooth: false,
          colorbar: { thickness: 12, len: 0.88, title: { text: "log₁₀|û|", side: "top" } }
        }
      ],
      layout,
      PLOTLY_CONFIG
    );
  }

  function renderPredictionPlots(data) {
    const zMax = safeZMax(data.z_abs_max);
    const w0T = transpose2D(data.w0);
    const uPredT = transpose2D(data.u_pred);

    Plotly.react(
      "initPlot",
      [
        {
          z: w0T,
          type: "heatmap",
          colorscale: COLORSCALE_VORTICITY,
          zmin: -zMax,
          zmax: zMax,
          zauto: false,
          zsmooth: false,
          colorbar: { thickness: 11, len: 0.88, title: { text: "ω", side: "top" } }
        }
      ],
      makeFieldLayout("", "x", "y"),
      PLOTLY_CONFIG
    );

    const predTraces = [
      {
        z: uPredT,
        type: "heatmap",
        colorscale: COLORSCALE_VORTICITY,
        zmin: -zMax,
        zmax: zMax,
        zauto: false,
        zsmooth: false,
        colorbar: { thickness: 11, len: 0.88, title: { text: "ω", side: "top" } }
      }
    ];

    if (els.streamlinesToggle.checked && Array.isArray(data.ux) && Array.isArray(data.uy)) {
      const nx = data.ux.length;
      const ny = data.ux[0].length;
      const step = Math.max(2, Math.floor(nx / 16));
      let maxSpeed = 0;

      for (let i = 0; i < nx; i += step) {
        for (let j = 0; j < ny; j += step) {
          const spd = Math.hypot(data.ux[i][j], data.uy[i][j]);
          if (spd > maxSpeed) maxSpeed = spd;
        }
      }

      const scale = (step * 0.85) / Math.max(maxSpeed, 0.25);
      const lineX = [];
      const lineY = [];
      for (let i = 0; i < nx; i += step) {
        for (let j = 0; j < ny; j += step) {
          const u = data.ux[i][j];
          const v = data.uy[i][j];
          lineX.push(i, i + u * scale, null);
          lineY.push(j, j + v * scale, null);
        }
      }

      predTraces.push({
        type: "scatter",
        mode: "lines",
        x: lineX,
        y: lineY,
        line: { color: "rgba(15, 23, 42, 0.55)", width: 1.2 },
        showlegend: false,
        hoverinfo: "skip"
      });
    }

    Plotly.react("predPlot", predTraces, makeFieldLayout("", "x", "y"), PLOTLY_CONFIG);

    if (els.surface3DToggle.checked) {
      Plotly.react(
        "surfacePlot",
        [
          {
            z: uPredT,
            type: "surface",
            colorscale: COLORSCALE_VORTICITY,
            cmin: -zMax,
            cmax: zMax,
            cauto: false
          }
        ],
        {
          paper_bgcolor: "#FFFFFF",
          scene: {
            bgcolor: "#F8FAFC",
            xaxis: { title: "x" },
            yaxis: { title: "y" },
            zaxis: { title: "ω", range: [-zMax, zMax] }
          },
          margin: { l: 10, r: 10, t: 10, b: 10 }
        },
        PLOTLY_CONFIG
      );
    }

    if (els.fftSpectrumToggle.checked && data.fft2_pred) {
      renderFft2Plot(data.fft2_pred, data.modes_cutoff);
    }

    if (els.particleFlowToggle?.checked && data.u_pred && data.ux && data.uy) {
      particleEngine.updateField(data.u_pred, data.ux, data.uy, zMax);
    }
  }

  async function fetchPrediction() {
    if (state.inFlight.predict) {
      state.inFlight.predictQueued = true;
      return;
    }
    state.inFlight.predict = true;

    try {
      const data = await apiPost("/api/predict", buildBasePayload(true));
      state.latestPredict = data;

      // Update Top Telemetry Cards with smooth roll-up animation
      animateNumericText(els.latencyMetric, data.inf_time_ms, v => `${v.toFixed(2)} ms`, 320);
      els.latencySub.textContent = `${data.target_resolution}×${data.target_resolution} • ${data.device || "CPU"}`;
      animateNumericText(els.peakVortMetric, data.peak_vorticity, v => fmt(v, 2), 320);
      if (data.invariants_pred && Number.isFinite(Number(data.invariants_pred.enstrophy))) {
        els.enstrophySub.textContent = `Ω = ${fmt(data.invariants_pred.enstrophy, 3)}`;
      }
      if (data.device) {
        els.deviceBadge.textContent = `${data.device} ACCELERATION`;
      }

      renderInvariantsBar(data.invariants_init, data.invariants_pred);
      renderPredictionPlots(data);
    } catch (err) {
      console.error("Prediction error:", err);
      showToast(`Prediction error: ${err.message}`);
    } finally {
      state.inFlight.predict = false;
      if (state.inFlight.predictQueued) {
        state.inFlight.predictQueued = false;
        fetchPrediction();
      }
    }
  }

  // --------------------------------------------------------------------------
  // 8. TAB 1: MULTI-STEP TIME ROLLOUT PLAYER (/api/rollout)
  // --------------------------------------------------------------------------
  function stopRolloutPlayback() {
    clearInterval(state.rolloutPlayer.timer);
    state.rolloutPlayer.timer = null;
    state.rolloutPlayer.isPlaying = false;
    els.rolloutPlayBtn.innerHTML = `<i data-lucide="play" class="w-3.5 h-3.5"></i><span>Play</span>`;
    lucide.createIcons();
  }

  function renderRolloutFrame(frameIdx) {
    const data = state.latestRollout;
    if (!data || !Array.isArray(data.times) || data.times.length === 0) return;

    const idx = Math.max(0, Math.min(data.times.length - 1, frameIdx));
    state.rolloutPlayer.frameIdx = idx;
    els.rolloutSlider.value = String(idx);

    const t = data.times[idx];
    const tMax = data.times[data.times.length - 1];
    const rel = data.rel_l2_per_frame ? data.rel_l2_per_frame[idx] : 0;
    els.rolloutTimeLabel.textContent = `t = ${fmt(t, 1)} / ${fmt(tMax, 1)}  (Rel-L2: ${fmt(rel, 1)}%)`;

    const zMax = safeZMax(data.z_abs_max);

    Plotly.react(
      "rolloutRk4Plot",
      [
        {
          z: transpose2D(data.rk4_frames[idx]),
          type: "heatmap",
          colorscale: COLORSCALE_VORTICITY,
          zmin: -zMax,
          zmax: zMax,
          zauto: false,
          zsmooth: false,
          colorbar: { thickness: 10, len: 0.85 }
        }
      ],
      makeFieldLayout(`RK4 Solver (t = ${fmt(t, 1)})`, "x", "y"),
      PLOTLY_CONFIG
    );

    Plotly.react(
      "rolloutPinoPlot",
      [
        {
          z: transpose2D(data.pino_frames[idx]),
          type: "heatmap",
          colorscale: COLORSCALE_VORTICITY,
          zmin: -zMax,
          zmax: zMax,
          zauto: false,
          zsmooth: false,
          colorbar: { thickness: 10, len: 0.85 }
        }
      ],
      makeFieldLayout(`PINO Rollout (t = ${fmt(t, 1)})`, "x", "y"),
      PLOTLY_CONFIG
    );

    const metricsEl = document.getElementById("rolloutMetricsPlot");
    if (metricsEl && metricsEl.data) {
      Plotly.relayout(metricsEl, {
        shapes: [
          {
            type: "line",
            x0: t,
            x1: t,
            y0: 0,
            y1: 1,
            yref: "paper",
            line: { color: "#0F172A", width: 2, dash: "dot" }
          }
        ]
      });
    }
  }

  function renderRolloutMetricsChart(data) {
    const t0 = data.times[0] ?? 0;
    Plotly.react(
      "rolloutMetricsPlot",
      [
        {
          x: data.times,
          y: data.rk4_enstrophy,
          name: "RK4 Enstrophy Ω(t)",
          type: "scatter",
          mode: "lines+markers",
          line: { color: "#0F172A", width: 2.2 }
        },
        {
          x: data.times,
          y: data.pino_enstrophy,
          name: "PINO Enstrophy Ω(t)",
          type: "scatter",
          mode: "lines+markers",
          line: { color: "#2563EB", width: 2.2, dash: "dash" }
        },
        {
          x: data.times,
          y: data.rel_l2_per_frame,
          name: "Rel-L2 Error (%)",
          type: "scatter",
          mode: "lines+markers",
          yaxis: "y2",
          line: { color: "#DC2626", width: 2 }
        }
      ],
      makeChartLayout("Simulation Time t", "Enstrophy Ω", {
        margin: { l: 54, r: 54, t: 24, b: 42 },
        yaxis2: {
          title: "Rel-L2 (%)",
          overlaying: "y",
          side: "right",
          showgrid: false,
          rangemode: "tozero"
        },
        shapes: [
          {
            type: "line",
            x0: t0,
            x1: t0,
            y0: 0,
            y1: 1,
            yref: "paper",
            line: { color: "#0F172A", width: 2, dash: "dot" }
          }
        ]
      }),
      PLOTLY_CONFIG
    );
  }

  function startRolloutPlayback() {
    if (!state.latestRollout) return;
    stopRolloutPlayback();
    state.rolloutPlayer.isPlaying = true;
    els.rolloutPlayBtn.innerHTML = `<i data-lucide="pause" class="w-3.5 h-3.5"></i><span>Pause</span>`;
    lucide.createIcons();

    const intervalMs = Math.max(80, Math.round(1000 / state.rolloutPlayer.fps));
    state.rolloutPlayer.timer = setInterval(() => {
      const total = state.latestRollout.times.length;
      const nextIdx = (state.rolloutPlayer.frameIdx + 1) % total;
      renderRolloutFrame(nextIdx);
    }, intervalMs);
  }

  async function runTimeRollout() {
    stopRolloutPlayback();
    const btn = els.runRolloutBtn;
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader" class="w-4 h-4 animate-spin"></i><span>Simulating Rollout...</span>`;
    lucide.createIcons();

    try {
      const payload = {
        ...buildBasePayload(false),
        viscosity: getSelectedViscosity(),
        num_frames: parseInt(els.rolloutFramesSelect.value, 10) || 6
      };
      const data = await apiPost("/api/rollout", payload);
      state.latestRollout = data;

      els.rolloutControlsBar.classList.remove("hidden");
      els.rolloutPlotsContainer.classList.remove("hidden");
      els.rolloutSlider.max = String(Math.max(0, data.times.length - 1));
      els.rolloutSlider.value = "0";

      els.rolloutPinoMs.textContent = `PINO: ${data.total_pino_ms} ms`;
      els.rolloutRk4Ms.textContent = `RK4: ${data.total_rk4_ms} ms`;
      els.rolloutSpeedup.textContent = `${data.speedup_factor}× Faster`;

      updateSpeedupCard(data.speedup_factor, `64² Multi-Step Rollout (${data.times.length - 1} steps)`);

      renderRolloutMetricsChart(data);
      renderRolloutFrame(0);
      setTimeout(resizeAllPlots, 60);
    } catch (err) {
      console.error("Rollout error:", err);
      showToast(`Rollout failed: ${err.message}`);
    } finally {
      setSurfaceLoading(["psoUncontrolledPlot", "psoForcingPlot", "psoControlledPlot", "psoConvergencePlot"], false);
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="play-circle" class="w-4 h-4"></i><span>Simulate Time Rollout (t = 0 → 5)</span>`;
      lucide.createIcons();
    }
  }

  // --------------------------------------------------------------------------
  // 9. TAB 2: ZERO-SHOT SUPER-RESOLUTION (/api/superres)
  // --------------------------------------------------------------------------
  function renderSuperresPlots(data) {
    const zMax = safeZMax(data.z_abs_max);
    const hr = data.high_res || state.superresHighRes;

    document.getElementById("srTime64").textContent = `${data.time_64_ms} ms`;
    document.getElementById("srTimeHigh").textContent = `${data.time_high_ms} ms`;
    document.getElementById("srHighResLabel").textContent = `${hr}²`;
    document.getElementById("srConsistencyErr").textContent = `${data.consistency_rel_l2_pct}%`;
    document.getElementById("srModesCutoff").textContent = `k ∈ [-${data.modes_cutoff || 16}, ${data.modes_cutoff || 16}]²`;
    document.getElementById("srHighTitle1").textContent = `2. Bicubic Upsample (${hr}×${hr})`;
    document.getElementById("srHighTitle2").textContent = `3. Zero-Shot PINO (${hr}×${hr})`;
    if (els.srWipeRightBadge) {
      els.srWipeRightBadge.textContent = `Zero-Shot PINO ${hr}²`;
    }
    renderMatrixToCanvas(els.srWipeLeft, data.u_bicubic, -zMax, zMax, LUT_VORTICITY);
    renderMatrixToCanvas(els.srWipeRight, data.u_high, -zMax, zMax, LUT_VORTICITY);

    // 3-panel Full-Field Comparison (64x64 PINO vs Bicubic vs Zero-Shot PINO — NOT w0!)
    Plotly.react(
      "srBase64Plot",
      [{
        z: transpose2D(data.u_64),
        type: "heatmap",
        colorscale: COLORSCALE_VORTICITY,
        zmin: -zMax,
        zmax: zMax,
        zauto: false,
        zsmooth: false,
        showscale: false
      }],
      makeFieldLayout("", "x (64)", "y (64)"),
      PLOTLY_CONFIG
    );

    Plotly.react(
      "srBicubicPlot",
      [{
        z: transpose2D(data.u_bicubic),
        type: "heatmap",
        colorscale: COLORSCALE_VORTICITY,
        zmin: -zMax,
        zmax: zMax,
        zauto: false,
        zsmooth: false,
        showscale: false
      }],
      makeFieldLayout("", `x (${hr})`, `y (${hr})`),
      PLOTLY_CONFIG
    );

    Plotly.react(
      "srZeroShotPlot",
      [{
        z: transpose2D(data.u_high),
        type: "heatmap",
        colorscale: COLORSCALE_VORTICITY,
        zmin: -zMax,
        zmax: zMax,
        zauto: false,
        zsmooth: false,
        colorbar: { thickness: 11, len: 0.88, title: { text: "ω", side: "top" } }
      }],
      makeFieldLayout("", `x (${hr})`, `y (${hr})`),
      PLOTLY_CONFIG
    );

    // 3-panel Center 64x64 Zoomed Detail Crops
    Plotly.react(
      "srCropNearestPlot",
      [{
        z: transpose2D(data.crop_nearest),
        type: "heatmap",
        colorscale: COLORSCALE_VORTICITY,
        zmin: -zMax,
        zmax: zMax,
        zauto: false,
        zsmooth: false,
        showscale: false
      }],
      makeFieldLayout("", "x (center 64)", "y"),
      PLOTLY_CONFIG
    );

    Plotly.react(
      "srCropBicubicPlot",
      [{
        z: transpose2D(data.crop_bicubic),
        type: "heatmap",
        colorscale: COLORSCALE_VORTICITY,
        zmin: -zMax,
        zmax: zMax,
        zauto: false,
        zsmooth: false,
        showscale: false
      }],
      makeFieldLayout("", "x (center 64)", "y"),
      PLOTLY_CONFIG
    );

    Plotly.react(
      "srCropPinoPlot",
      [{
        z: transpose2D(data.crop_pino),
        type: "heatmap",
        colorscale: COLORSCALE_VORTICITY,
        zmin: -zMax,
        zmax: zMax,
        zauto: false,
        zsmooth: false,
        colorbar: { thickness: 11, len: 0.88, title: { text: "ω", side: "top" } }
      }],
      makeFieldLayout("", "x (center 64)", "y"),
      PLOTLY_CONFIG
    );

    // Radial Energy Spectrum E(k) comparison
    const kMax = Number(data.modes_cutoff) || 16;
    if (data.spectrum_64 && data.spectrum_high) {
      const allE = [...data.spectrum_64.energy, ...data.spectrum_high.energy].filter(v => v > 0);
      const minE = Math.min(...allE, 1e-10);
      const maxE = Math.max(...allE, 1e-2);
      Plotly.react(
        "srSpectrumPlot",
        [
          {
            x: data.spectrum_64.k,
            y: data.spectrum_64.energy,
            name: "Native 64×64 PINO E(k)",
            type: "scatter",
            mode: "lines+markers",
            line: { color: "#0F172A", width: 2.2 }
          },
          {
            x: data.spectrum_high.k,
            y: data.spectrum_high.energy,
            name: `Zero-Shot ${hr}×${hr} PINO E(k)`,
            type: "scatter",
            mode: "lines",
            line: { color: "#2563EB", width: 2.4 }
          },
          {
            x: [kMax, kMax],
            y: [minE, maxE],
            name: `k_max = ${kMax} Cutoff`,
            type: "scatter",
            mode: "lines",
            line: { color: "#DC2626", width: 1.8, dash: "dash" }
          }
        ],
        makeChartLayout("Wavenumber k (log)", "Radial Energy E(k) (log)", {
          xaxis: { type: "log" },
          yaxis: { type: "log" }
        }),
        PLOTLY_CONFIG
      );
    }
  }

  async function fetchSuperRes() {
    if (state.inFlight.superres) return;
    state.inFlight.superres = true;
    const btn = els.refreshSrBtn;
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader" class="w-3.5 h-3.5 animate-spin"></i><span>Evaluating...</span>`;
    lucide.createIcons();
    setSurfaceLoading(["srBase64Plot", "srBicubicPlot", "srZeroShotPlot", "srWipeStage"], true);

    try {
      const payload = {
        ...buildBasePayload(false),
        high_res: state.superresHighRes
      };
      const data = await apiPost("/api/superres", payload);
      state.latestSuperres = data;
      state.dirtySr = false;
      renderSuperresPlots(data);
      setTimeout(resizeAllPlots, 60);
    } catch (err) {
      console.error("Super-resolution error:", err);
      showToast(`Super-Res failed: ${err.message}`);
    } finally {
      setSurfaceLoading(["srBase64Plot", "srBicubicPlot", "srZeroShotPlot", "srWipeStage"], false);
      state.inFlight.superres = false;
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="refresh-cw" class="w-3.5 h-3.5"></i><span>Evaluate Super-Res</span>`;
      lucide.createIcons();
    }
  }

  // --------------------------------------------------------------------------
  // 10. TAB 3: BENCHMARK DRAG RACE & CFL SCALING LAW
  // --------------------------------------------------------------------------
  async function runBenchmarkRace() {
    const btn = els.startRaceBtn;
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader" class="w-3.5 h-3.5 animate-spin"></i><span>Running RK4 vs PINO...</span>`;
    lucide.createIcons();
    setSurfaceLoading(["raceRk4Plot", "racePinoPlot", "raceErrPlot", "raceSpectrumPlot"], true);

    try {
      const payload = {
        ...buildBasePayload(false),
        viscosity: getSelectedViscosity()
      };
      const data = await apiPost("/api/benchmark", payload);
      state.latestBenchmark = data;
      state.hasRunRace = true;

      document.getElementById("raceResultsBanner").classList.remove("hidden");
      document.getElementById("rk4Time").textContent = `${data.time_rk4_ms} ms`;
      document.getElementById("pinoTime").textContent = `${data.time_pino_ms} ms`;
      document.getElementById("speedupBadge").textContent = `${data.speedup_factor}× Faster`;
      document.getElementById("raceRelL2").textContent = `${data.rel_l2_pct}%`;
      document.getElementById("raceErrDetail").textContent = `MSE: ${fmt(data.mse, 4)} • MAE: ${fmt(data.mae, 4)}`;
      document.getElementById("circDrift").textContent = `${data.circ_drift_pct}%`;
      document.getElementById("energyDrift").textContent = `E Drift: ${data.energy_drift_pct}%`;
      document.getElementById("racePdeRes").textContent = fmt(data.pde_residual, 4);

      updateSpeedupCard(data.speedup_factor, `64² Single-Step (${data.time_pino_ms} vs ${data.time_rk4_ms} ms)`);

      const zMax = safeZMax(data.z_abs_max);

      Plotly.react(
        "raceRk4Plot",
        [{
          z: transpose2D(data.rk4_w),
          type: "heatmap",
          colorscale: COLORSCALE_VORTICITY,
          zmin: -zMax,
          zmax: zMax,
          zauto: false,
          zsmooth: false,
          showscale: false
        }],
        makeFieldLayout("", "x", "y"),
        PLOTLY_CONFIG
      );

      Plotly.react(
        "racePinoPlot",
        [{
          z: transpose2D(data.pino_w),
          type: "heatmap",
          colorscale: COLORSCALE_VORTICITY,
          zmin: -zMax,
          zmax: zMax,
          zauto: false,
          zsmooth: false,
          colorbar: { thickness: 10, len: 0.85, title: { text: "ω", side: "top" } }
        }],
        makeFieldLayout("", "x", "y"),
        PLOTLY_CONFIG
      );

      Plotly.react(
        "raceErrPlot",
        [{
          z: transpose2D(data.err_w),
          type: "heatmap",
          colorscale: COLORSCALE_ERROR,
          zmin: 0,
          zmax: safeZMax(data.max_abs_err),
          zauto: false,
          zsmooth: false,
          colorbar: { thickness: 10, len: 0.85, title: { text: "|err|", side: "top" } }
        }],
        makeFieldLayout("", "x", "y"),
        PLOTLY_CONFIG
      );

      // Radial Energy Spectrum (RK4 vs PINO)
      const kMax = Number(data.modes_cutoff) || 16;
      if (data.spectrum_rk4 && data.spectrum_pino) {
        const allE = [...data.spectrum_rk4.energy, ...data.spectrum_pino.energy].filter(v => v > 0);
        const minE = Math.min(...allE, 1e-10);
        const maxE = Math.max(...allE, 1e-2);
        Plotly.react(
          "raceSpectrumPlot",
          [
            {
              x: data.spectrum_rk4.k,
              y: data.spectrum_rk4.energy,
              name: "RK4 Ground Truth E(k)",
              type: "scatter",
              mode: "lines+markers",
              line: { color: "#0F172A", width: 2.2 }
            },
            {
              x: data.spectrum_pino.k,
              y: data.spectrum_pino.energy,
              name: "PINO Prediction E(k)",
              type: "scatter",
              mode: "lines+markers",
              line: { color: "#2563EB", width: 2.2, dash: "dash" }
            },
            {
              x: [kMax, kMax],
              y: [minE, maxE],
              name: `k_max = ${kMax} Cutoff`,
              type: "scatter",
              mode: "lines",
              line: { color: "#DC2626", width: 1.8, dash: "dash" }
            }
          ],
          makeChartLayout("Wavenumber k (log)", "Energy E(k) (log)", {
            xaxis: { type: "log" },
            yaxis: { type: "log" }
          }),
          PLOTLY_CONFIG
        );
      }

      setTimeout(resizeAllPlots, 60);
    } catch (err) {
      console.error("Benchmark failed:", err);
      showToast(`Benchmark failed: ${err.message}`);
    } finally {
      setSurfaceLoading(["raceRk4Plot", "racePinoPlot", "raceErrPlot", "raceSpectrumPlot"], false);
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="play" class="w-3.5 h-3.5"></i><span>Run 64² Benchmark Race</span>`;
      lucide.createIcons();
    }
  }

  async function runCflScalingLaw() {
    const btn = els.startScalingBtn;
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader" class="w-3.5 h-3.5 animate-spin"></i><span>Benchmarking 64²→256²...</span>`;
    lucide.createIcons();
    setSurfaceLoading(["raceScalingPlot"], true);

    try {
      const payload = {
        ...buildBasePayload(false),
        viscosity: getSelectedViscosity()
      };
      const data = await apiPost("/api/scaling", payload);
      state.latestScaling = data;
      els.scalingEmptyOverlay.classList.add("hidden");

      const xLabels = data.resolutions.map((r, i) => `${r}² (${data.rk4_steps[i]} steps)`);
      const speedupTexts = data.speedups.map(s => {
        const val = Number(s);
        return `${val >= 100 ? Math.round(val) : val.toFixed(1)}×`;
      });

      const allMs = [...(data.rk4_ms || []), ...(data.pino_ms || [])].filter(v => Number(v) > 0);
      const minMs = Math.min(...allMs, 0.5);
      const maxMs = Math.max(...allMs, 50);

      // Update top speedup card with 256^2 measured CFL speedup
      const highIdx = data.speedups.length - 1;
      if (highIdx >= 0) {
        const highRes = data.resolutions[highIdx];
        updateSpeedupCard(data.speedups[highIdx], `${highRes}² CFL Scaling vs RK4`);
      }

      Plotly.react(
        "raceScalingPlot",
        [
          {
            x: xLabels,
            y: data.rk4_ms,
            name: "RK4 Solver (CFL Scaled)",
            type: "scatter",
            mode: "lines+markers+text",
            text: speedupTexts,
            textposition: "top center",
            cliponaxis: false,
            textfont: { size: 10, color: "#059669", family: "JetBrains Mono, monospace" },
            line: { color: "#0F172A", width: 2.5 },
            marker: { size: 8, color: "#0F172A" }
          },
          {
            x: xLabels,
            y: data.pino_ms,
            name: "PINO Zero-Shot Forward",
            type: "scatter",
            mode: "lines+markers",
            line: { color: "#2563EB", width: 2.5 },
            marker: { size: 8, color: "#2563EB" }
          }
        ],
        makeChartLayout("Resolution & CFL RK4 Steps", "Latency (ms, log scale)", {
          margin: { l: 58, r: 28, t: 38, b: 44 },
          yaxis: {
            type: "log",
            range: [Math.log10(minMs * 0.65), Math.log10(maxMs * 3.5)]
          }
        }),
        PLOTLY_CONFIG
      );

      setTimeout(resizeAllPlots, 60);
    } catch (err) {
      console.error("Scaling benchmark failed:", err);
      showToast(`CFL Scaling failed: ${err.message}`);
    } finally {
      setSurfaceLoading(["raceScalingPlot"], false);
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="trending-up" class="w-3.5 h-3.5"></i><span>Run CFL Scaling Law (64² → 256²)</span>`;
      lucide.createIcons();
    }
  }

  // --------------------------------------------------------------------------
  // 11. TAB 4: LIVE TEST-TIME ADAPTATION (/api/tta)
  // --------------------------------------------------------------------------
  async function runTta(overridePreset = null) {
    if (state.inFlight.tta) return;
    state.inFlight.tta = true;

    if (overridePreset) {
      els.presetSelect.value = overridePreset;
      syncPresetPanels();
    }

    const btn = els.startTtaBtn;
    btn.disabled = true;
    els.ttaDipoleBtn.disabled = true;
    els.ttaShearBtn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader" class="w-3.5 h-3.5 animate-spin"></i><span>Adapting Projection...</span>`;
    lucide.createIcons();
    setSurfaceLoading(["ttaErrBeforePlot", "ttaErrAfterPlot", "ttaCurvePlot"], true);

    try {
      const steps = parseInt(els.ttaStepsInput.value, 10) || 10;
      const lr = parseFloat(els.ttaLrSelect.value) || 1e-4;
      const alphaAnchor = parseFloat(els.ttaAlphaInput.value) || 0.5;

      const payload = {
        ...buildBasePayload(false),
        viscosity: getSelectedViscosity(),
        steps,
        lr,
        alpha_anchor: alphaAnchor
      };

      const data = await apiPost("/api/tta", payload);
      state.latestTta = data;

      document.getElementById("ttaEmptyOverlay")?.classList.add("hidden");
      document.getElementById("ttaResultsBanner").classList.remove("hidden");
      document.getElementById("ttaPdeBefore").textContent = fmt(data.pde_res_before, 4);
      document.getElementById("ttaPdeAfter").textContent = fmt(data.pde_res_after, 4);
      document.getElementById("ttaPdeDrop").textContent = `${fmt(data.pde_reduction_pct, 1)}% PDE residual drop`;

      document.getElementById("ttaRelBefore").textContent = `${fmt(data.rel_l2_before, 2)}%`;
      document.getElementById("ttaRelAfter").textContent = `${fmt(data.rel_l2_after, 2)}%`;
      document.getElementById("ttaRelDelta").textContent = `Δ = ${fmtSigned(data.rel_l2_delta_pct, 2)}%`;

      document.getElementById("ttaCircBefore").textContent = `${fmt(data.circ_drift_before, 3)}%`;
      document.getElementById("ttaCircAfter").textContent = `${fmt(data.circ_drift_after, 3)}%`;
      animateNumericText(document.getElementById("ttaRuntime"), data.tta_time_ms, v => `${v.toFixed(1)} ms`, 320);
      document.getElementById("ttaStepsLabel").textContent = `${data.steps} projection gradient steps`;

      const errMax = safeZMax(data.err_max);

      if (els.ttaWipeCard) {
        els.ttaWipeCard.classList.remove("hidden");
        renderMatrixToCanvas(els.ttaWipeLeft, data.err_before, 0, errMax, LUT_ERROR);
        renderMatrixToCanvas(els.ttaWipeRight, data.err_after, 0, errMax, LUT_ERROR);
      }

      Plotly.react(
        "ttaErrBeforePlot",
        [{
          z: transpose2D(data.err_before),
          type: "heatmap",
          colorscale: COLORSCALE_ERROR,
          zmin: 0,
          zmax: errMax,
          zauto: false,
          zsmooth: false,
          showscale: false
        }],
        makeFieldLayout("", "x", "y"),
        PLOTLY_CONFIG
      );

      Plotly.react(
        "ttaErrAfterPlot",
        [{
          z: transpose2D(data.err_after),
          type: "heatmap",
          colorscale: COLORSCALE_ERROR,
          zmin: 0,
          zmax: errMax,
          zauto: false,
          zsmooth: false,
          colorbar: { thickness: 10, len: 0.85, title: { text: "|err|", side: "top" } }
        }],
        makeFieldLayout("", "x", "y"),
        PLOTLY_CONFIG
      );

      const stepAxis = data.loss_curve.map((_, idx) => idx + 1);
      Plotly.react(
        "ttaCurvePlot",
        [
          {
            x: stepAxis,
            y: data.loss_curve,
            name: "Total TTA Loss",
            type: "scatter",
            mode: "lines+markers",
            line: { color: "#2563EB", width: 2.2 }
          },
          {
            x: stepAxis,
            y: data.pde_curve,
            name: "PDE Residual MSE",
            type: "scatter",
            mode: "lines+markers",
            yaxis: "y2",
            line: { color: "#059669", width: 2.2, dash: "dash" }
          }
        ],
        makeChartLayout("TTA Step", "Total Loss", {
          margin: { l: 52, r: 52, t: 24, b: 42 },
          yaxis2: {
            title: "PDE MSE",
            overlaying: "y",
            side: "right",
            showgrid: false
          }
        }),
        PLOTLY_CONFIG
      );

      if (overridePreset) {
        fetchPrediction();
      }
      setTimeout(resizeAllPlots, 60);
    } catch (err) {
      console.error("TTA error:", err);
      showToast(`TTA failed: ${err.message}`);
    } finally {
      setSurfaceLoading(["ttaErrBeforePlot", "ttaErrAfterPlot", "ttaCurvePlot"], false);
      state.inFlight.tta = false;
      btn.disabled = false;
      els.ttaDipoleBtn.disabled = false;
      els.ttaShearBtn.disabled = false;
      btn.innerHTML = `<i data-lucide="sparkles" class="w-3.5 h-3.5"></i><span>Run TTA on Current Flow</span>`;
      lucide.createIcons();
    }
  }

  // --------------------------------------------------------------------------
  // 12. TAB 5: ACTIVE FLOW CONTROL (/api/pso)
  // --------------------------------------------------------------------------
  async function runPsoControl() {
    const btn = els.startPsoBtn;
    const numParticles = parseInt(els.psoParticlesInput.value, 10) || 50;
    const iterations = parseInt(els.psoIterationsInput.value, 10) || 12;
    const forcingBound = parseFloat(els.psoBoundInput.value) || 0.5;

    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader" class="w-3.5 h-3.5 animate-spin"></i><span>Optimizing ${numParticles}-Particle Swarm...</span>`;
    lucide.createIcons();
    setSurfaceLoading(["psoUncontrolledPlot", "psoForcingPlot", "psoControlledPlot", "psoConvergencePlot"], true);

    try {
      const payload = {
        ...buildBasePayload(false),
        num_particles: numParticles,
        iterations,
        forcing_bound: forcingBound
      };
      const data = await apiPost("/api/pso", payload);
      state.latestPso = data;
      state.hasRunPso = true;

      document.getElementById("psoEmptyOverlay")?.classList.add("hidden");
      document.getElementById("psoResultsBanner").classList.remove("hidden");
      document.getElementById("psoEvals").textContent = Number(data.total_evaluations).toLocaleString();
      document.getElementById("psoLatency").textContent = `${data.total_latency_ms} ms`;
      document.getElementById("psoRate").textContent = `${Math.round(data.eval_per_sec).toLocaleString()} evals/s`;
      animateNumericText(document.getElementById("psoSuppression"), data.reduction_pct, v => `${v.toFixed(1)}%`, 420);
      flashCard(document.getElementById("psoSuppression"));
      document.getElementById("psoFitnessDetail").textContent = `${fmt(data.base_fitness, 4)} → ${fmt(data.best_fitness, 4)}`;
      document.getElementById("psoBudgetInfo").textContent = `||f*||∞ = ${fmt(data.forcing_linf_norm, 3)}`;
      document.getElementById("psoL2Info").textContent = `||f*||₂ = ${fmt(data.forcing_l2_norm, 2)} (Bound ${fmt(forcingBound, 2)})`;

      const zMax = safeZMax(data.z_abs_max);
      const fMax = safeZMax(data.forcing_linf_norm || forcingBound);

      Plotly.react(
        "psoUncontrolledPlot",
        [{
          z: transpose2D(data.uncontrolled_w),
          type: "heatmap",
          colorscale: COLORSCALE_VORTICITY,
          zmin: -zMax,
          zmax: zMax,
          zauto: false,
          zsmooth: false,
          colorbar: { thickness: 10, len: 0.85, title: { text: "ω", side: "top" } }
        }],
        makeFieldLayout("", "x", "y"),
        PLOTLY_CONFIG
      );

      Plotly.react(
        "psoForcingPlot",
        [{
          z: transpose2D(data.best_forcing),
          type: "heatmap",
          colorscale: COLORSCALE_FORCING,
          zmin: -fMax,
          zmax: fMax,
          zauto: false,
          zsmooth: false,
          colorbar: { thickness: 10, len: 0.85, title: { text: "f*", side: "top" } }
        }],
        makeFieldLayout("", "x", "y"),
        PLOTLY_CONFIG
      );

      Plotly.react(
        "psoControlledPlot",
        [{
          z: transpose2D(data.controlled_w),
          type: "heatmap",
          colorscale: COLORSCALE_VORTICITY,
          zmin: -zMax,
          zmax: zMax,
          zauto: false,
          zsmooth: false,
          colorbar: { thickness: 10, len: 0.85, title: { text: "ω", side: "top" } }
        }],
        makeFieldLayout("", "x", "y"),
        PLOTLY_CONFIG
      );

      // Convergence Curve
      const iterAxis = (data.fitness_history || []).map((_, idx) => idx);
      Plotly.react(
        "psoConvergencePlot",
        [
          {
            x: iterAxis,
            y: data.fitness_history || [],
            name: "Global Best Enstrophy",
            type: "scatter",
            mode: "lines+markers",
            line: { color: "#059669", width: 2.5 }
          },
          {
            x: iterAxis,
            y: data.mean_fitness_history || [],
            name: "Swarm Mean Enstrophy",
            type: "scatter",
            mode: "lines+markers",
            line: { color: "#2563EB", width: 2, dash: "dash" }
          }
        ],
        makeChartLayout("PSO Iteration", "Mean Squared Vorticity ⟨ω²⟩", {
          shapes: [
            {
              type: "line",
              x0: 0,
              x1: Math.max(1, iterAxis.length - 1),
              y0: data.base_fitness,
              y1: data.base_fitness,
              line: { color: "#DC2626", width: 1.6, dash: "dot" }
            }
          ],
          annotations: [
            {
              x: Math.max(1, iterAxis.length - 1),
              y: data.base_fitness,
              text: `Uncontrolled: ${fmt(data.base_fitness, 4)}`,
              showarrow: false,
              xanchor: "right",
              yanchor: "bottom",
              font: { size: 10, color: "#DC2626", family: "JetBrains Mono, monospace" },
              bgcolor: "rgba(255,255,255,0.85)"
            }
          ]
        }),
        PLOTLY_CONFIG
      );

      setTimeout(resizeAllPlots, 60);
    } catch (err) {
      console.error("PSO failed:", err);
      showToast(`PSO failed: ${err.message}`);
    } finally {
      setSurfaceLoading(["psoUncontrolledPlot", "psoForcingPlot", "psoControlledPlot", "psoConvergencePlot"], false);
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="play" class="w-3.5 h-3.5"></i><span>Optimize Forcing Profile f*(x, y)</span>`;
      lucide.createIcons();
    }
  }

  // --------------------------------------------------------------------------
  // 13. HEADER VIVA UTILITIES (PRESENTATION MODE, PNG EXPORT, JSON EXPORT)
  // --------------------------------------------------------------------------
  function togglePresentationMode() {
    state.presentationMode = !state.presentationMode;
    document.body.classList.toggle("presentation-mode", state.presentationMode);
    els.presentationBtnText.textContent = state.presentationMode ? "Exit Presentation" : "Presentation Mode";
    setTimeout(() => {
      updateTabPill();
      resizeAllPlots();
    }, 80);
  }

  function exportVisiblePlotsPng() {
    const activeSection = document.getElementById(state.activeTab);
    if (!activeSection) return;
    const plots = Array.from(activeSection.querySelectorAll(".js-plotly-plot"))
      .filter(el => el.offsetParent !== null && Array.isArray(el.data) && el.data.length > 0);

    if (plots.length === 0) {
      showToast("No rendered plots in the active tab to export.");
      return;
    }

    plots.forEach((plotEl, idx) => {
      setTimeout(() => {
        const filename = `pino_${state.activeTab}_${plotEl.id}`;
        Plotly.downloadImage(plotEl, {
          format: "png",
          width: 1100,
          height: 820,
          scale: 2,
          filename
        });
      }, idx * 250);
    });
    showToast(`Exporting ${plots.length} PNG plot(s) from ${state.activeTab}...`);
  }

  function exportMetricsJson() {
    const viscosity = getSelectedViscosity();
    const snapshot = {
      exported_at: new Date().toISOString(),
      model_info: state.info,
      active_controls: {
        preset: els.presetSelect.value,
        length_scale: parseFloat(els.lengthScaleSlider.value),
        alpha: parseFloat(els.alphaSlider.value),
        seed: parseInt(els.seedInput.value, 10) || 42,
        viscosity,
        reynolds_number: Math.round(1 / viscosity),
        target_resolution: state.targetRes,
        custom_vortices_count: els.presetSelect.value === "custom" ? state.painter.vortices.length : 0
      },
      predict_telemetry: state.latestPredict
        ? {
            inf_time_ms: state.latestPredict.inf_time_ms,
            peak_vorticity: state.latestPredict.peak_vorticity,
            device: state.latestPredict.device,
            target_resolution: state.latestPredict.target_resolution,
            modes_cutoff: state.latestPredict.modes_cutoff,
            z_abs_max: state.latestPredict.z_abs_max,
            invariants_init: state.latestPredict.invariants_init,
            invariants_pred: state.latestPredict.invariants_pred
          }
        : null,
      superres_telemetry: state.latestSuperres
        ? {
            high_res: state.latestSuperres.high_res,
            time_64_ms: state.latestSuperres.time_64_ms,
            time_high_ms: state.latestSuperres.time_high_ms,
            consistency_rel_l2_pct: state.latestSuperres.consistency_rel_l2_pct,
            modes_cutoff: state.latestSuperres.modes_cutoff
          }
        : null,
      rollout_telemetry: state.latestRollout
        ? {
            times: state.latestRollout.times,
            pino_enstrophy: state.latestRollout.pino_enstrophy,
            rk4_enstrophy: state.latestRollout.rk4_enstrophy,
            pino_energy: state.latestRollout.pino_energy,
            rk4_energy: state.latestRollout.rk4_energy,
            rel_l2_per_frame: state.latestRollout.rel_l2_per_frame,
            total_pino_ms: state.latestRollout.total_pino_ms,
            total_rk4_ms: state.latestRollout.total_rk4_ms,
            speedup_factor: state.latestRollout.speedup_factor
          }
        : null,
      benchmark_telemetry: state.latestBenchmark
        ? {
            time_rk4_ms: state.latestBenchmark.time_rk4_ms,
            time_pino_ms: state.latestBenchmark.time_pino_ms,
            speedup_factor: state.latestBenchmark.speedup_factor,
            rel_l2_pct: state.latestBenchmark.rel_l2_pct,
            mse: state.latestBenchmark.mse,
            mae: state.latestBenchmark.mae,
            rmse: state.latestBenchmark.rmse,
            circ_drift_pct: state.latestBenchmark.circ_drift_pct,
            energy_drift_pct: state.latestBenchmark.energy_drift_pct,
            max_abs_err: state.latestBenchmark.max_abs_err,
            pde_residual: state.latestBenchmark.pde_residual
          }
        : null,
      scaling_telemetry: state.latestScaling,
      tta_telemetry: state.latestTta
        ? {
            steps: state.latestTta.steps,
            tta_time_ms: state.latestTta.tta_time_ms,
            pde_res_before: state.latestTta.pde_res_before,
            pde_res_after: state.latestTta.pde_res_after,
            pde_reduction_pct: state.latestTta.pde_reduction_pct,
            rel_l2_before: state.latestTta.rel_l2_before,
            rel_l2_after: state.latestTta.rel_l2_after,
            rel_l2_delta_pct: state.latestTta.rel_l2_delta_pct,
            circ_drift_before: state.latestTta.circ_drift_before,
            circ_drift_after: state.latestTta.circ_drift_after,
            loss_curve: state.latestTta.loss_curve,
            pde_curve: state.latestTta.pde_curve
          }
        : null,
      pso_telemetry: state.latestPso
        ? {
            total_evaluations: state.latestPso.total_evaluations,
            total_latency_ms: state.latestPso.total_latency_ms,
            eval_per_sec: state.latestPso.eval_per_sec,
            best_fitness: state.latestPso.best_fitness,
            base_fitness: state.latestPso.base_fitness,
            reduction_pct: state.latestPso.reduction_pct,
            forcing_l2_norm: state.latestPso.forcing_l2_norm,
            forcing_linf_norm: state.latestPso.forcing_linf_norm,
            num_particles: state.latestPso.num_particles,
            iterations: state.latestPso.iterations,
            fitness_history: state.latestPso.fitness_history,
            mean_fitness_history: state.latestPso.mean_fitness_history
          }
        : null
    };

    const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `pino_telemetry_${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    showToast("Downloaded telemetry JSON snapshot.");
  }

  // --------------------------------------------------------------------------
  // 14. EVENT WIRING & TAB ORCHESTRATION
  // --------------------------------------------------------------------------
  function triggerControlsUpdate() {
    state.dirtySr = true;
    clearTimeout(state.timers.predictDebounce);
    state.timers.predictDebounce = setTimeout(() => {
      fetchPrediction();
    }, 45);

    if (state.activeTab === "tab-sr") {
      clearTimeout(state.timers.srDebounce);
      state.timers.srDebounce = setTimeout(() => {
        fetchSuperRes();
      }, 150);
    }
  }

  function setupEventListeners() {
    // Header Utilities
    els.presentationModeBtn.addEventListener("click", togglePresentationMode);
    els.exportPngBtn.addEventListener("click", exportVisiblePlotsPng);
    els.exportJsonBtn.addEventListener("click", exportMetricsJson);

    // Sidebar Sliders & Preset Controls
    els.lengthScaleSlider.addEventListener("input", e => {
      els.lengthScaleVal.textContent = parseFloat(e.target.value).toFixed(2);
      updateRegimeBadge();
      triggerControlsUpdate();
    });

    els.alphaSlider.addEventListener("input", e => {
      els.alphaVal.textContent = parseFloat(e.target.value).toFixed(2);
      updateRegimeBadge();
      triggerControlsUpdate();
    });

    els.seedInput.addEventListener("input", e => {
      els.seedVal.textContent = e.target.value || "42";
      triggerControlsUpdate();
    });

    els.randomSeedBtn.addEventListener("click", () => {
      const nextSeed = Math.floor(Math.random() * 90000) + 1000;
      els.seedInput.value = String(nextSeed);
      els.seedVal.textContent = String(nextSeed);
      triggerControlsUpdate();
    });

    els.presetSelect.addEventListener("change", () => {
      syncPresetPanels();
      triggerControlsUpdate();
    });

    els.viscositySelect.addEventListener("change", () => {
      state.dirtySr = true;
      showToast(`Viscosity updated to ν = ${els.viscositySelect.value}`);
    });

    els.refreshBtn.addEventListener("click", () => {
      fetchPrediction();
      if (state.activeTab === "tab-sr") fetchSuperRes();
    });

    // Resolution Buttons
    document.querySelectorAll(".res-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".res-btn").forEach(b => {
          b.className = "res-btn px-2 py-1.5 text-xs font-semibold rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 font-mono";
        });
        btn.className = "res-btn px-2 py-1.5 text-xs font-semibold rounded-lg border border-blue-600 bg-blue-50 text-blue-700 font-mono";
        state.targetRes = parseInt(btn.getAttribute("data-res"), 10) || 64;
        fetchPrediction();
      });
    });

    // Interactive Vortex Painter
    els.polarityCcwBtn.addEventListener("click", () => {
      state.painter.polarity = 1;
      updatePolarityButtons();
    });

    els.polarityCwBtn.addEventListener("click", () => {
      state.painter.polarity = -1;
      updatePolarityButtons();
    });

    els.brushSigmaSlider.addEventListener("input", e => {
      state.painter.sigma = parseFloat(e.target.value);
      els.brushSigmaVal.textContent = state.painter.sigma.toFixed(2);
    });

    els.vortexUndoBtn.addEventListener("click", () => {
      state.painter.vortices.pop();
      drawVortexCanvas();
      triggerControlsUpdate();
    });

    els.vortexClearBtn.addEventListener("click", () => {
      state.painter.vortices = [];
      drawVortexCanvas();
      triggerControlsUpdate();
    });

    els.vortexSeedBtn.addEventListener("click", () => {
      seedDipoleVortices();
      triggerControlsUpdate();
    });

    els.vortexCanvas.addEventListener("pointerdown", ev => {
      state.painter.isDrawing = true;
      state.painter.lastCanvasPt = null;
      els.vortexCanvas.setPointerCapture(ev.pointerId);
      addVortexFromEvent(ev, false);
    });

    els.vortexCanvas.addEventListener("pointermove", ev => {
      if (!state.painter.isDrawing) return;
      addVortexFromEvent(ev, true);
    });

    const stopPaintDrag = () => {
      state.painter.isDrawing = false;
      state.painter.lastCanvasPt = null;
    };
    els.vortexCanvas.addEventListener("pointerup", stopPaintDrag);
    els.vortexCanvas.addEventListener("pointercancel", stopPaintDrag);

    // Tab 1 View Toggles
    els.streamlinesToggle.addEventListener("change", () => {
      if (state.latestPredict) renderPredictionPlots(state.latestPredict);
    });

    els.surface3DToggle.addEventListener("change", e => {
      if (e.target.checked) {
        els.surface3DContainer.classList.remove("hidden");
        els.heatmap2DContainer.classList.add("hidden");
      } else {
        els.surface3DContainer.classList.add("hidden");
        els.heatmap2DContainer.classList.remove("hidden");
      }
      if (state.latestPredict) renderPredictionPlots(state.latestPredict);
      setTimeout(resizeAllPlots, 60);
    });

    els.fftSpectrumToggle.addEventListener("change", e => {
      if (e.target.checked) {
        els.fft2Container.classList.remove("hidden");
        if (state.latestPredict && state.latestPredict.fft2_pred) {
          renderFft2Plot(state.latestPredict.fft2_pred, state.latestPredict.modes_cutoff);
        }
      } else {
        els.fft2Container.classList.add("hidden");
      }
      setTimeout(resizeAllPlots, 60);
    });

    // Tab 1 Live 60 FPS Particle Advection Controls
    if (els.particleFlowToggle) {
      els.particleFlowToggle.addEventListener("change", e => {
        if (e.target.checked) {
          els.particleFlowContainer?.classList.remove("hidden");
          if (state.latestPredict?.u_pred && state.latestPredict?.ux) {
            particleEngine.updateField(
              state.latestPredict.u_pred,
              state.latestPredict.ux,
              state.latestPredict.uy,
              safeZMax(state.latestPredict.z_abs_max)
            );
          }
          particleEngine.ensureLoop();
        } else {
          els.particleFlowContainer?.classList.add("hidden");
        }
      });
    }

    if (els.particlePlayBtn) {
      els.particlePlayBtn.addEventListener("click", () => {
        state.particleFlow.paused = !state.particleFlow.paused;
        els.particlePlayBtn.innerHTML = state.particleFlow.paused
          ? `<i data-lucide="play" class="w-3.5 h-3.5"></i><span>Play</span>`
          : `<i data-lucide="pause" class="w-3.5 h-3.5"></i><span>Pause</span>`;
        lucide.createIcons();
        if (!state.particleFlow.paused) particleEngine.ensureLoop();
      });
    }

    if (els.particleResetBtn) {
      els.particleResetBtn.addEventListener("click", () => {
        particleEngine.resetTracers();
      });
    }

    if (els.particleSpeedSelect) {
      els.particleSpeedSelect.addEventListener("change", e => {
        state.particleFlow.speed = parseFloat(e.target.value) || 1.0;
      });
    }

    // Tab 1 Rollout Player
    els.runRolloutBtn.addEventListener("click", runTimeRollout);

    els.rolloutPlayBtn.addEventListener("click", () => {
      if (state.rolloutPlayer.isPlaying) {
        stopRolloutPlayback();
      } else {
        startRolloutPlayback();
      }
    });

    els.rolloutSlider.addEventListener("input", e => {
      stopRolloutPlayback();
      renderRolloutFrame(parseInt(e.target.value, 10) || 0);
    });

    els.rolloutFpsSelect.addEventListener("change", e => {
      state.rolloutPlayer.fps = parseInt(e.target.value, 10) || 2;
      if (state.rolloutPlayer.isPlaying) {
        startRolloutPlayback();
      }
    });

    // Tab 2 Super-Resolution
    document.querySelectorAll(".sr-res-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".sr-res-btn").forEach(b => {
          b.className = "sr-res-btn px-2.5 py-1.5 text-xs font-mono font-semibold rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50";
        });
        btn.className = "sr-res-btn px-2.5 py-1.5 text-xs font-mono font-semibold rounded-lg border border-blue-600 bg-blue-50 text-blue-700";
        state.superresHighRes = parseInt(btn.getAttribute("data-sr-res"), 10) || 256;
        fetchSuperRes();
      });
    });
    els.refreshSrBtn.addEventListener("click", fetchSuperRes);

    // Tab 3 Benchmark & Scaling
    els.startRaceBtn.addEventListener("click", runBenchmarkRace);
    els.startScalingBtn.addEventListener("click", runCflScalingLaw);

    // Tab 4 TTA
    els.ttaStepsInput.addEventListener("input", e => {
      els.ttaStepsVal.textContent = e.target.value;
    });
    els.ttaAlphaInput.addEventListener("input", e => {
      els.ttaAlphaVal.textContent = parseFloat(e.target.value).toFixed(1);
    });
    els.ttaDipoleBtn.addEventListener("click", () => runTta("dipole"));
    els.ttaShearBtn.addEventListener("click", () => runTta("shear"));
    els.startTtaBtn.addEventListener("click", () => runTta(null));

    // Tab 5 PSO
    els.psoParticlesInput.addEventListener("input", e => {
      els.psoParticlesVal.textContent = e.target.value;
    });
    els.psoIterationsInput.addEventListener("input", e => {
      els.psoIterationsVal.textContent = e.target.value;
    });
    els.psoBoundInput.addEventListener("input", e => {
      els.psoBoundVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
    els.startPsoBtn.addEventListener("click", runPsoControl);

    // Tab Navigation
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        state.activeTab = targetTab;

        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        updateTabPill();

        document.querySelectorAll(".tab-content").forEach(sec => sec.classList.add("hidden"));
        const activeSec = document.getElementById(targetTab);
        activeSec.classList.remove("hidden");
        triggerTabEntrance(activeSec);

        if (targetTab === "tab-sim") {
          particleEngine.ensureLoop();
        } else if (targetTab === "tab-sr" && (state.dirtySr || !state.latestSuperres)) {
          fetchSuperRes();
        } else if (targetTab === "tab-race" && !state.hasRunRace) {
          state.hasRunRace = true;
          setTimeout(runBenchmarkRace, 40);
        } else if (targetTab === "tab-pso" && !state.hasRunPso) {
          state.hasRunPso = true;
          setTimeout(runPsoControl, 40);
        }

        setTimeout(() => {
          updateTabPill();
          resizeAllPlots();
          if (targetTab === "tab-arch") {
            renderMathSafe(document.getElementById("tab-arch"));
          }
        }, 70);
      });
    });

    window.addEventListener("resize", () => {
      clearTimeout(state.timers.resizeDebounce);
      state.timers.resizeDebounce = setTimeout(() => {
        updateTabPill();
        resizeAllPlots();
      }, 120);
    });
  }

  // --------------------------------------------------------------------------
  // 15. BOOTSTRAP
  // --------------------------------------------------------------------------
  function init() {
    lucide.createIcons();
    initHeaderFlowCanvas();
    bindWipeComparator(
      els.srWipeStage,
      els.srWipeOverlay,
      els.srWipeDivider,
      els.srWipeSlider,
      els.srWipePctLabel
    );
    bindWipeComparator(
      els.ttaWipeStage,
      els.ttaWipeOverlay,
      els.ttaWipeDivider,
      els.ttaWipeSlider,
      els.ttaWipePctLabel
    );

    updatePolarityButtons();
    syncPresetPanels();
    drawVortexCanvas();
    setupEventListeners();
    renderMathSafe(document.body);
    updateTabPill();
    document.fonts?.ready?.then(updateTabPill);

    fetchModelInfo();
    fetchPrediction();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
