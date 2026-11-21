/**
 * LexiFix AI — Modern Interactive Application Logic
 * Colour Palette:
 * 1. Eclipse Plum (#2B1C33) - 50%
 * 2. Pebble Silk (#D9D3CC) - 30% (Prominently Integrated)
 * 3. Amber Flare (#FF8C12) - 20%
 */

// Application State
const state = {
  currentView: 'studio',
  theme: localStorage.getItem('lexifix_theme') || 'dark',
  lastResult: null,
  isProcessing: false,
  csvFiles: [],
  currentCsvFile: '',
  csvData: { headers: [], rows: [] },
  csvFilteredRows: [],
  csvSortCol: null,
  csvSortAsc: true,
  vocabulary: { size: 0, tokens: 0, top: [] },
  sysInfo: {}
};

// DOM Cache
const dom = {};

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  cacheDOMElements();
  bindEvents();
  applyTheme(state.theme);
  loadSystemInfo();
  loadVocabularyStats();
  loadCsvFileList();
  updateInputStats();
  
  // Explain default word
  explainWord('recieve');
});

function cacheDOMElements() {
  // Navigation
  dom.navButtons = document.querySelectorAll('[data-nav]');
  dom.views = {
    studio: document.getElementById('view-studio'),
    explorer: document.getElementById('view-explorer'),
    batch: document.getElementById('view-batch'),
    explain: document.getElementById('view-explain'),
    settings: document.getElementById('view-settings')
  };

  // Studio Elements
  dom.inputText = document.getElementById('input-text');
  dom.expectedText = document.getElementById('expected-text');
  dom.expectedContainer = document.getElementById('expected-container');
  dom.btnToggleExpected = document.getElementById('btn-toggle-expected');
  dom.btnRemoveExpected = document.getElementById('btn-remove-expected');
  dom.btnClearInput = document.getElementById('btn-clear-input');
  dom.btnCorrect = document.getElementById('btn-correct');
  dom.correctSpinner = document.getElementById('correct-spinner');
  dom.correctBtnText = document.getElementById('correct-btn-text');
  dom.statWords = document.getElementById('stat-words');
  dom.statChars = document.getElementById('stat-chars');
  dom.detectedBadge = document.getElementById('detected-count-badge');
  dom.liveDetectedPreview = document.getElementById('live-detected-preview');
  dom.sliderMaxDistance = document.getElementById('slider-max-distance');
  dom.labelMaxDistance = document.getElementById('label-max-distance');
  dom.sampleChips = document.querySelectorAll('.sample-chip');
  dom.modelChecks = document.querySelectorAll('.model-check');
  dom.btnToggleCustom = document.getElementById('btn-toggle-custom-model');
  dom.customModelDrawer = document.getElementById('custom-model-drawer');
  dom.inputCustomModel = document.getElementById('input-custom-model');

  // Baseline Output
  dom.baselineOutputDiff = document.getElementById('baseline-output-diff');
  dom.baselineOutputClean = document.getElementById('baseline-output-clean');
  dom.baselineOutputExplain = document.getElementById('baseline-output-explain');
  dom.baselineMetricsBar = document.getElementById('baseline-metrics-bar');
  dom.mBaselineAcc = document.getElementById('m-baseline-acc');
  dom.mBaselinePrec = document.getElementById('m-baseline-prec');
  dom.mBaselineRec = document.getElementById('m-baseline-rec');
  dom.mBaselineF1 = document.getElementById('m-baseline-f1');
  dom.baselineLatency = document.getElementById('baseline-latency');
  dom.llmCardsGrid = document.getElementById('llm-cards-grid');

  // Export Bar
  dom.exportBar = document.getElementById('export-bar');
  dom.btnSaveRunCsv = document.getElementById('btn-save-run-csv');
  dom.btnJumpExplorer = document.getElementById('btn-jump-explorer');

  // Explorer Elements
  dom.csvFileSelect = document.getElementById('csv-file-select');
  dom.btnRefreshCsv = document.getElementById('btn-refresh-csv');
  dom.csvSearchInput = document.getElementById('csv-search-input');
  dom.csvRowCounter = document.getElementById('csv-row-counter');
  dom.benchmarkThead = document.getElementById('benchmark-thead');
  dom.benchmarkTbody = document.getElementById('benchmark-tbody');
  dom.kpiTotalCases = document.getElementById('kpi-total-cases');
  dom.kpiBaselineF1 = document.getElementById('kpi-baseline-f1');
  dom.kpiLlmF1 = document.getElementById('kpi-llm-f1');
  dom.kpiAccuracyComp = document.getElementById('kpi-accuracy-comp');

  // Batch Elements
  dom.batchCasesFile = document.getElementById('batch-cases-file');
  dom.batchOutputFilename = document.getElementById('batch-output-filename');
  dom.batchCheckT5 = document.getElementById('batch-check-t5');
  dom.batchCheckFlan = document.getElementById('batch-check-flan');
  dom.btnStartBatch = document.getElementById('btn-start-batch');
  dom.batchProgressBox = document.getElementById('batch-progress-box');
  dom.batchProgressStatus = document.getElementById('batch-progress-status');
  dom.batchProgressPct = document.getElementById('batch-progress-pct');
  dom.batchProgressBar = document.getElementById('batch-progress-bar');
  dom.batchLogOutput = document.getElementById('batch-log-output');

  // Sandbox Elements
  dom.explainWordInput = document.getElementById('explain-word-input');
  dom.btnExplainWord = document.getElementById('btn-explain-word');
  dom.explainVocabStatus = document.getElementById('explain-vocab-status');
  dom.explainBestWord = document.getElementById('explain-best-word');
  dom.explainDistance = document.getElementById('explain-distance');
  dom.explainConfidence = document.getElementById('explain-confidence');
  dom.explainCandidatesList = document.getElementById('explain-candidates-list');
  dom.statVocabSize = document.getElementById('stat-vocab-size');
  dom.statCorpusTokens = document.getElementById('stat-corpus-tokens');
  dom.topLexiconChips = document.getElementById('top-lexicon-chips');

  // Modals & General
  dom.btnThemeToggle = document.getElementById('btn-theme-toggle');
  dom.themeIconSun = document.getElementById('theme-icon-sun');
  dom.themeIconMoon = document.getElementById('theme-icon-moon');
  dom.btnHelpModal = document.getElementById('btn-help-modal');
  dom.helpModal = document.getElementById('help-modal');
  dom.btnCloseHelp = document.getElementById('btn-close-help');
  dom.caseModal = document.getElementById('case-modal');
  dom.btnCloseModal = document.getElementById('btn-close-modal');
  dom.modalBodyContent = document.getElementById('modal-body-content');
  dom.settingsThemeSelect = document.getElementById('settings-theme-select');
  dom.toastContainer = document.getElementById('toast-container');
}

function bindEvents() {
  // Navigation tabs
  dom.navButtons.forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.nav));
  });

  // Theme Toggle
  dom.btnThemeToggle.addEventListener('click', toggleTheme);
  dom.settingsThemeSelect.addEventListener('change', (e) => applyTheme(e.target.value));

  // Studio Events
  dom.inputText.addEventListener('input', updateInputStats);
  dom.btnClearInput.addEventListener('click', () => {
    dom.inputText.value = '';
    updateInputStats();
  });

  dom.btnToggleExpected.addEventListener('click', () => {
    dom.expectedContainer.classList.remove('hidden');
    dom.btnToggleExpected.classList.add('hidden');
  });

  dom.btnRemoveExpected.addEventListener('click', () => {
    dom.expectedContainer.classList.add('hidden');
    dom.expectedText.value = '';
    dom.btnToggleExpected.classList.remove('hidden');
  });

  dom.sliderMaxDistance.addEventListener('input', (e) => {
    dom.labelMaxDistance.textContent = `${e.target.value} edit${e.target.value > 1 ? 's' : ''}`;
  });

  dom.btnToggleCustom.addEventListener('click', () => {
    dom.customModelDrawer.classList.toggle('hidden');
    if (!dom.customModelDrawer.classList.contains('hidden')) {
      dom.inputCustomModel.focus();
    }
  });

  dom.sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      dom.inputText.value = chip.dataset.sample;
      updateInputStats();
      showToast('Loaded sample text', 'info');
    });
  });

  dom.btnCorrect.addEventListener('click', triggerCorrection);

  // View Mode Switcher buttons
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.view-mode-btn');
    if (btn) {
      const target = btn.dataset.target;
      const mode = btn.dataset.mode;
      const group = btn.parentElement.querySelectorAll('.view-mode-btn');
      group.forEach(b => {
        b.classList.remove('active', 'bg-[#FF8C12]', 'text-[#2B1C33]', 'font-bold');
        b.classList.add('text-[#D9D3CC]');
      });
      btn.classList.add('active', 'bg-[#FF8C12]', 'text-[#2B1C33]', 'font-bold');
      btn.classList.remove('text-[#D9D3CC]');
      toggleCardViewMode(target, mode);
    }

    // Copy to clipboard buttons
    const copyBtn = e.target.closest('.btn-copy');
    if (copyBtn) {
      const target = copyBtn.dataset.copyTarget;
      copyCardOutput(target);
    }
  });

  // Export Bar
  dom.btnSaveRunCsv.addEventListener('click', saveRunToCsv);
  dom.btnJumpExplorer.addEventListener('click', () => switchView('explorer'));

  // CSV Explorer Events
  dom.csvFileSelect.addEventListener('change', (e) => {
    if (e.target.value) loadCsvContent(e.target.value);
  });
  dom.btnRefreshCsv.addEventListener('click', () => {
    loadCsvFileList(state.currentCsvFile);
  });
  dom.csvSearchInput.addEventListener('input', filterCsvTable);

  // Batch Evaluator Events
  dom.btnStartBatch.addEventListener('click', startBatchEvaluation);

  // Explainability Sandbox
  dom.btnExplainWord.addEventListener('click', () => {
    explainWord(dom.explainWordInput.value.trim());
  });
  dom.explainWordInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') explainWord(dom.explainWordInput.value.trim());
  });

  // Modals & Shortcuts
  dom.btnHelpModal.addEventListener('click', () => dom.helpModal.classList.add('active'));
  dom.btnCloseHelp.addEventListener('click', () => dom.helpModal.classList.remove('active'));
  dom.btnCloseModal.addEventListener('click', () => dom.caseModal.classList.remove('active'));

  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (state.currentView === 'studio') triggerCorrection();
    }
    if (e.altKey && e.key === '1') switchView('studio');
    if (e.altKey && e.key === '2') switchView('explorer');
    if (e.altKey && e.key === '3') switchView('batch');
    if (e.altKey && e.key === '4') switchView('explain');
    if (e.altKey && e.key === '5') switchView('settings');
    if (e.altKey && (e.key === 't' || e.key === 'T')) toggleTheme();
    if (e.key === 'Escape') {
      dom.helpModal.classList.remove('active');
      dom.caseModal.classList.remove('active');
    }
    if (e.key === '?' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
      dom.helpModal.classList.add('active');
    }
  });
}

// ----------------- Navigation & Theme -----------------

function switchView(viewName) {
  if (!dom.views[viewName]) return;
  state.currentView = viewName;

  dom.navButtons.forEach(btn => {
    const isActive = btn.dataset.nav === viewName;
    btn.classList.toggle('active', isActive);
    btn.classList.toggle('bg-[#FF8C12]/15', isActive);
    btn.classList.toggle('text-[#FF8C12]', isActive);
    btn.classList.toggle('border-[#FF8C12]/45', isActive);
    btn.classList.toggle('text-[#D9D3CC]', !isActive);
  });

  Object.entries(dom.views).forEach(([name, el]) => {
    el.classList.toggle('hidden', name !== viewName);
  });
}

function applyTheme(theme) {
  state.theme = theme;
  localStorage.setItem('lexifix_theme', theme);
  document.documentElement.setAttribute('data-theme', theme);
  if (dom.settingsThemeSelect) dom.settingsThemeSelect.value = theme;
  
  if (theme === 'dark') {
    dom.themeIconSun.classList.add('hidden');
    dom.themeIconMoon.classList.remove('hidden');
  } else {
    dom.themeIconSun.classList.remove('hidden');
    dom.themeIconMoon.classList.add('hidden');
  }
}

function toggleTheme() {
  applyTheme(state.theme === 'dark' ? 'light' : 'dark');
}

// ----------------- Toast Notifications -----------------

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${message}</span>
  `;
  dom.toastContainer.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ----------------- Live Studio Logic -----------------

function updateInputStats() {
  const text = dom.inputText.value;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const chars = text.length;

  dom.statWords.textContent = `${words} word${words !== 1 ? 's' : ''}`;
  dom.statChars.textContent = `${chars} character${chars !== 1 ? 's' : ''}`;

  // Quick offline vocabulary check for detected unknowns
  if (state.vocabulary && state.vocabulary.set && words > 0) {
    const tokens = text.match(/[A-Za-z]+(?:'[A-Za-z]+)?/g) || [];
    const unknowns = [...new Set(tokens.filter(t => !state.vocabulary.set.has(t.toLowerCase())))];
    if (unknowns.length > 0) {
      dom.detectedBadge.textContent = `${unknowns.length} error${unknowns.length > 1 ? 's' : ''} detected`;
      dom.detectedBadge.className = 'px-2 py-0.5 rounded-full bg-[#FF8C12]/20 text-[#FF8C12] text-[10px] font-mono border border-[#FF8C12]/40 font-bold';
      dom.liveDetectedPreview.textContent = `(${unknowns.slice(0, 3).join(', ')}${unknowns.length > 3 ? '...' : ''})`;
    } else {
      dom.detectedBadge.textContent = '0 errors detected';
      dom.detectedBadge.className = 'px-2 py-0.5 rounded-full bg-[#D9D3CC]/15 text-[#D9D3CC] text-[10px] font-mono border border-[#D9D3CC]/30 font-semibold';
      dom.liveDetectedPreview.textContent = '';
    }
  }
}

async function triggerCorrection() {
  const text = dom.inputText.value.trim();
  if (!text) {
    showToast('Please enter text to correct', 'error');
    return;
  }

  // Selected models
  const selectedModels = [];
  dom.modelChecks.forEach(ch => {
    if (ch.checked) selectedModels.push(ch.value);
  });
  const customModel = dom.inputCustomModel.value.trim();
  if (customModel && !selectedModels.includes(customModel)) {
    selectedModels.push(customModel);
  }

  const maxDistance = parseInt(dom.sliderMaxDistance.value, 10);
  const expectedText = dom.expectedText.value.trim();

  // Update UI to running state
  state.isProcessing = true;
  dom.btnCorrect.disabled = true;
  dom.correctSpinner.classList.remove('hidden');
  dom.correctBtnText.textContent = 'Processing...';

  renderShimmerSkeletons(selectedModels);

  try {
    const response = await fetch('/api/correct', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        models: selectedModels,
        max_distance: maxDistance,
        expected_text: expectedText
      })
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || 'Server error occurred');
    }

    const data = await response.json();
    state.lastResult = data;
    renderResults(data);
    dom.exportBar.classList.remove('hidden');
    showToast('Correction & comparison complete', 'success');
  } catch (error) {
    console.error(error);
    showToast(`Error: ${error.message}`, 'error');
  } finally {
    state.isProcessing = false;
    dom.btnCorrect.disabled = false;
    dom.correctSpinner.classList.add('hidden');
    dom.correctBtnText.textContent = '✦ Correct & Compare';
  }
}

function renderShimmerSkeletons(selectedModels) {
  dom.baselineOutputDiff.innerHTML = `
    <div class="space-y-2 w-full py-1">
      <div class="skeleton h-4 w-3/4"></div>
      <div class="skeleton h-4 w-1/2"></div>
    </div>
  `;
  dom.baselineOutputClean.innerHTML = '';
  dom.baselineOutputExplain.innerHTML = '';

  dom.llmCardsGrid.innerHTML = selectedModels.map(model => `
    <div class="surface-card p-5 space-y-4 border border-[#D9D3CC]/25 border-l-4 border-l-[#FF8C12]">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full bg-[#FF8C12] animate-pulse"></span>
          <h3 class="font-bold text-sm tracking-wide text-[#D9D3CC]">${escapeHtml(model)}</h3>
          <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-[#FF8C12]/20 text-[#FF8C12] font-semibold border border-[#FF8C12]/30">Processing...</span>
        </div>
      </div>
      <div class="p-4 bg-[#1F1325] rounded-xl border border-[#D9D3CC]/25 space-y-2">
        <div class="skeleton h-4 w-5/6"></div>
        <div class="skeleton h-4 w-2/3"></div>
      </div>
    </div>
  `).join('');
}

function renderResults(data) {
  const original = data.input_text;
  const baseline = data.baseline_text;

  // 1. Baseline Diff & Clean Text
  dom.baselineOutputDiff.innerHTML = generateTokenDiffHtml(original, baseline);
  dom.baselineOutputClean.textContent = baseline;

  // 2. Baseline Explainability Chips prominently styled in Pebble Silk
  if (data.corrections && data.corrections.length > 0) {
    dom.baselineOutputExplain.innerHTML = `
      <div class="space-y-2">
        <span class="text-xs font-bold text-[#D9D3CC]">Token-Level Diagnostic Breakdown:</span>
        <div class="flex flex-wrap gap-2">
          ${data.corrections.map(c => `
            <div class="correction-pill" title="Edit distance: ${c.distance}, Confidence: ${(c.confidence * 100).toFixed(0)}%">
              <span class="text-[#F87171] line-through">${escapeHtml(c.original)}</span>
              <span class="text-[#D9D3CC]/60 font-bold">→</span>
              <span class="text-[#FF8C12] font-bold">${escapeHtml(c.corrected)}</span>
              <span class="text-[10px] text-[#D9D3CC] px-1.5 py-0.5 rounded bg-[#D9D3CC]/20 border border-[#D9D3CC]/30 font-bold">d=${c.distance}</span>
              <span class="text-[10px] text-[#D9D3CC] font-mono">${(c.confidence * 100).toFixed(0)}%</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  } else {
    dom.baselineOutputExplain.innerHTML = `<span class="text-xs text-[#D9D3CC]/80 font-medium">No spelling modifications detected in vocabulary.</span>`;
  }

  // 3. Baseline Metrics with Pebble Silk labels
  if (data.expected_text) {
    dom.baselineMetricsBar.classList.remove('hidden');
    dom.mBaselineAcc.textContent = `${(data.baseline_accuracy * 100).toFixed(1)}%`;
    dom.mBaselinePrec.textContent = data.baseline_precision.toFixed(2);
    dom.mBaselineRec.textContent = data.baseline_recall.toFixed(2);
    dom.mBaselineF1.textContent = data.baseline_f1.toFixed(2);
  } else {
    dom.baselineMetricsBar.classList.add('hidden');
  }
  dom.baselineLatency.textContent = `Latency: ${data.baseline_latency_ms || 0}ms`;

  // 4. Render LLM Output Cards with Pebble Silk Framing & Amber Flare Accents
  if (data.llm_results && data.llm_results.length > 0) {
    dom.llmCardsGrid.innerHTML = data.llm_results.map((llm, idx) => {
      const isOk = llm.status === 'ok';
      const cardId = `llm-card-${idx}`;
      return `
        <div id="${cardId}" class="surface-card p-5 space-y-4 border border-[#D9D3CC]/25 border-l-4 border-l-[#FF8C12]">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div class="flex items-center gap-2.5">
              <span class="w-3 h-3 rounded-full ${isOk ? 'bg-[#FF8C12] shadow shadow-[#FF8C12]/50' : 'bg-rose-500'}"></span>
              <h3 class="font-bold text-sm tracking-wide text-[#D9D3CC]">${escapeHtml(llm.model)}</h3>
              <span class="text-[10px] font-mono px-2 py-0.5 rounded ${isOk ? 'bg-[#D9D3CC]/15 text-[#D9D3CC] border border-[#D9D3CC]/30 font-semibold' : 'bg-rose-500/20 text-rose-300'}">
                ${isOk ? 'Transformer' : 'Error'}
              </span>
              ${isOk ? `<span class="text-[10px] font-mono text-[#D9D3CC] bg-[#D9D3CC]/10 px-2 py-0.5 rounded border border-[#D9D3CC]/20">${llm.latency_ms || 0}ms</span>` : ''}
            </div>

            ${isOk ? `
              <div class="flex items-center gap-2">
                <div class="bg-[#221528] p-0.5 rounded-lg border border-[#D9D3CC]/30 flex text-xs">
                  <button class="view-mode-btn active px-2.5 py-1 rounded-md text-[#2B1C33] bg-[#FF8C12] font-bold" data-target="llm-${idx}" data-mode="diff">Visual Diff</button>
                  <button class="view-mode-btn px-2.5 py-1 rounded-md text-[#D9D3CC] hover:text-white" data-target="llm-${idx}" data-mode="clean">Clean Text</button>
                </div>

                <button class="btn-copy silk-btn p-1.5 rounded-lg text-[#D9D3CC] hover:text-[#FF8C12] transition-colors" data-copy-target="llm-${idx}" title="Copy output">
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                </button>
              </div>
            ` : ''}
          </div>

          <div class="p-4 bg-[#1F1325] rounded-xl border border-[#D9D3CC]/25 min-h-[64px]">
            ${isOk ? `
              <div id="llm-${idx}-output-diff" class="diff-container text-[#D9D3CC]">
                ${generateTokenDiffHtml(original, llm.text)}
              </div>
              <div id="llm-${idx}-output-clean" class="hidden text-[#D9D3CC] text-sm font-sans leading-relaxed">${escapeHtml(llm.text)}</div>
            ` : `
              <div class="text-rose-400 text-xs font-mono p-2 bg-rose-950/30 rounded border border-rose-900/50">
                ${escapeHtml(llm.error || 'Failed to generate output from model')}
              </div>
            `}
          </div>

          ${(isOk && data.expected_text) ? `
            <div class="pt-2 border-t border-[#D9D3CC]/20 flex flex-wrap items-center justify-between text-xs text-[#D9D3CC] font-mono">
              <div class="flex items-center gap-4">
                <span>Accuracy: <strong class="text-[#FF8C12]">${(llm.accuracy * 100).toFixed(1)}%</strong></span>
                <span>Precision: <strong class="text-[#D9D3CC]">${llm.precision.toFixed(2)}</strong></span>
                <span>Recall: <strong class="text-[#D9D3CC]">${llm.recall.toFixed(2)}</strong></span>
                <span>F1 Score: <strong class="text-[#FF8C12]">${llm.f1.toFixed(2)}</strong></span>
              </div>
            </div>
          ` : ''}
        </div>
      `;
    }).join('');
  } else {
    dom.llmCardsGrid.innerHTML = '';
  }
}

function toggleCardViewMode(target, mode) {
  if (target === 'baseline') {
    dom.baselineOutputDiff.classList.toggle('hidden', mode !== 'diff');
    dom.baselineOutputClean.classList.toggle('hidden', mode !== 'clean');
    dom.baselineOutputExplain.classList.toggle('hidden', mode !== 'explain');
  } else if (target.startsWith('llm-')) {
    const diffEl = document.getElementById(`${target}-output-diff`);
    const cleanEl = document.getElementById(`${target}-output-clean`);
    if (diffEl && cleanEl) {
      diffEl.classList.toggle('hidden', mode !== 'diff');
      cleanEl.classList.toggle('hidden', mode !== 'clean');
    }
  }
}

function copyCardOutput(target) {
  let textToCopy = '';
  if (target === 'baseline') {
    textToCopy = dom.baselineOutputClean.textContent;
  } else if (target.startsWith('llm-')) {
    const cleanEl = document.getElementById(`${target}-output-clean`);
    if (cleanEl) textToCopy = cleanEl.textContent;
  }
  if (textToCopy) {
    navigator.clipboard.writeText(textToCopy);
    showToast('Copied to clipboard', 'success');
  }
}

async function saveRunToCsv() {
  if (!state.lastResult) return;
  try {
    const res = await fetch('/api/csv/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.lastResult)
    });
    if (!res.ok) throw new Error('Failed to save to CSV');
    const result = await res.json();
    showToast(`Saved to ${result.filename}!`, 'success');
    loadCsvFileList(result.filename);
  } catch (err) {
    showToast(`Save failed: ${err.message}`, 'error');
  }
}

// ----------------- Visual Diff Algorithm -----------------

function generateTokenDiffHtml(source, target) {
  if (!source || !target) return escapeHtml(target || '');
  if (source.trim() === target.trim()) return `<span>${escapeHtml(target)}</span>`;

  // Tokenize preserving whitespace & punctuation
  const tokenize = (text) => text.split(/(\s+|[.,!?;:"]+)/).filter(Boolean);
  const srcTokens = tokenize(source);
  const tgtTokens = tokenize(target);

  let sIdx = 0;
  let tIdx = 0;
  let html = '';

  while (sIdx < srcTokens.length || tIdx < tgtTokens.length) {
    const s = srcTokens[sIdx];
    const t = tgtTokens[tIdx];

    if (s === t) {
      html += `<span>${escapeHtml(t)}</span>`;
      sIdx++;
      tIdx++;
    } else if (s && !t) {
      // Deletion
      html += `<span class="diff-del">${escapeHtml(s)}</span>`;
      sIdx++;
    } else if (!s && t) {
      // Insertion in Amber Flare with Pebble Silk contrast
      html += `<span class="diff-ins">${escapeHtml(t)}</span>`;
      tIdx++;
    } else {
      const nextMatchInTgt = tgtTokens.indexOf(s, tIdx);
      const nextMatchInSrc = srcTokens.indexOf(t, sIdx);

      if (nextMatchInTgt !== -1 && (nextMatchInTgt - tIdx <= 2)) {
        while (tIdx < nextMatchInTgt) {
          html += `<span class="diff-ins">${escapeHtml(tgtTokens[tIdx])}</span>`;
          tIdx++;
        }
      } else if (nextMatchInSrc !== -1 && (nextMatchInSrc - sIdx <= 2)) {
        while (sIdx < nextMatchInSrc) {
          html += `<span class="diff-del">${escapeHtml(srcTokens[sIdx])}</span>`;
          sIdx++;
        }
      } else {
        html += `<span class="diff-del">${escapeHtml(s)}</span><span class="diff-ins">${escapeHtml(t)}</span>`;
        sIdx++;
        tIdx++;
      }
    }
  }

  return html;
}

// ----------------- CSV Explorer -----------------

async function loadCsvFileList(autoSelectFile) {
  try {
    const res = await fetch('/api/csv/list');
    const files = await res.json();
    state.csvFiles = files;

    dom.csvFileSelect.innerHTML = files.map(f => `
      <option value="${escapeHtml(f.name)}">${escapeHtml(f.name)} (${f.rows} rows • ${f.size_kb} KB)</option>
    `).join('');

    // Also populate Batch Cases selector
    dom.batchCasesFile.innerHTML = files.filter(f => f.name.endsWith('.csv')).map(f => `
      <option value="${escapeHtml(f.name)}">${escapeHtml(f.name)} (${f.rows} cases)</option>
    `).join('');

    const targetFile = autoSelectFile || (files[0] ? files[0].name : '');
    if (targetFile) {
      dom.csvFileSelect.value = targetFile;
      loadCsvContent(targetFile);
    }
  } catch (err) {
    console.error(err);
  }
}

async function loadCsvContent(filename) {
  state.currentCsvFile = filename;
  try {
    const res = await fetch(`/api/csv/read?file=${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error('Failed to read CSV');
    const data = await res.json();
    state.csvData = data;
    state.csvFilteredRows = [...data.rows];
    state.csvSortCol = null;

    updateKpiCards(data);
    renderCsvTable();
  } catch (err) {
    showToast(`Failed to load CSV: ${err.message}`, 'error');
  }
}

function updateKpiCards(data) {
  const rows = data.rows || [];
  dom.kpiTotalCases.textContent = rows.length;

  if (rows.length === 0) {
    dom.kpiBaselineF1.textContent = '—';
    dom.kpiLlmF1.textContent = '—';
    dom.kpiAccuracyComp.textContent = '—';
    return;
  }

  // Calculate Mean Baseline F1
  const bF1s = rows.map(r => parseFloat(r.baseline_f1)).filter(v => !isNaN(v));
  if (bF1s.length > 0) {
    const mean = bF1s.reduce((a, b) => a + b, 0) / bF1s.length;
    dom.kpiBaselineF1.textContent = mean.toFixed(3);
  } else {
    dom.kpiBaselineF1.textContent = '—';
  }

  // Calculate Mean LLM F1
  let llmCol = 'llm_f1';
  if (!data.headers.includes(llmCol)) {
    const candidate = data.headers.find(h => h.endsWith('_f1') && h !== 'baseline_f1');
    if (candidate) llmCol = candidate;
  }
  const lF1s = rows.map(r => parseFloat(r[llmCol])).filter(v => !isNaN(v));
  if (lF1s.length > 0) {
    const mean = lF1s.reduce((a, b) => a + b, 0) / lF1s.length;
    dom.kpiLlmF1.textContent = mean.toFixed(3);
  } else {
    dom.kpiLlmF1.textContent = '—';
  }

  // Mean Word Accuracy
  const bAccs = rows.map(r => parseFloat(r.baseline_accuracy)).filter(v => !isNaN(v));
  if (bAccs.length > 0) {
    const mean = bAccs.reduce((a, b) => a + b, 0) / bAccs.length;
    dom.kpiAccuracyComp.textContent = `${(mean * 100).toFixed(1)}%`;
  } else {
    dom.kpiAccuracyComp.textContent = '—';
  }
}

function renderCsvTable() {
  const headers = state.csvData.headers || [];
  const rows = state.csvFilteredRows || [];

  dom.csvRowCounter.textContent = `Showing ${rows.length} of ${state.csvData.rows.length} rows`;

  if (headers.length === 0) {
    dom.benchmarkThead.innerHTML = `<tr><th>No Columns</th></tr>`;
    dom.benchmarkTbody.innerHTML = `<tr><td class="text-center py-6 text-[#D9D3CC]/60 font-medium">Empty file</td></tr>`;
    return;
  }

  // Render Table Headers with sort indicator
  dom.benchmarkThead.innerHTML = `<tr>${headers.map(h => `
    <th class="table-sort-header" data-col="${escapeHtml(h)}">
      <div class="flex items-center gap-1.5">
        <span>${escapeHtml(h)}</span>
        ${state.csvSortCol === h ? `<span>${state.csvSortAsc ? '↑' : '↓'}</span>` : ''}
      </div>
    </th>
  `).join('')}</tr>`;

  // Bind sort headers
  dom.benchmarkThead.querySelectorAll('.table-sort-header').forEach(th => {
    th.addEventListener('click', () => sortCsvByColumn(th.dataset.col));
  });

  // Render Rows in Pebble Silk
  if (rows.length === 0) {
    dom.benchmarkTbody.innerHTML = `<tr><td colspan="${headers.length}" class="text-center py-8 text-[#D9D3CC]/70 font-medium">No matching cases found</td></tr>`;
    return;
  }

  dom.benchmarkTbody.innerHTML = rows.map((row, rowIdx) => `
    <tr class="csv-row-item" data-row-idx="${rowIdx}">
      ${headers.map(h => {
        const val = row[h] || '';
        const isNumeric = !isNaN(parseFloat(val)) && isFinite(val);
        return `<td class="${isNumeric ? 'font-mono text-[#FF8C12] font-semibold' : 'text-[#D9D3CC]'}" title="${escapeHtml(val)}">${escapeHtml(val)}</td>`;
      }).join('')}
    </tr>
  `).join('');

  // Row Inspection Modal binding
  dom.benchmarkTbody.querySelectorAll('.csv-row-item').forEach(tr => {
    tr.addEventListener('click', () => {
      const idx = parseInt(tr.dataset.rowIdx, 10);
      inspectCsvRow(rows[idx]);
    });
  });
}

function filterCsvTable() {
  const query = dom.csvSearchInput.value.toLowerCase().trim();
  if (!query) {
    state.csvFilteredRows = [...state.csvData.rows];
  } else {
    state.csvFilteredRows = state.csvData.rows.filter(row => {
      return Object.values(row).some(val => String(val).toLowerCase().includes(query));
    });
  }
  renderCsvTable();
}

function sortCsvByColumn(col) {
  if (state.csvSortCol === col) {
    state.csvSortAsc = !state.csvSortAsc;
  } else {
    state.csvSortCol = col;
    state.csvSortAsc = true;
  }

  state.csvFilteredRows.sort((a, b) => {
    const valA = a[col] || '';
    const valB = b[col] || '';
    const numA = parseFloat(valA);
    const numB = parseFloat(valB);

    if (!isNaN(numA) && !isNaN(numB)) {
      return state.csvSortAsc ? numA - numB : numB - numA;
    }
    return state.csvSortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
  });

  renderCsvTable();
}

function inspectCsvRow(row) {
  dom.modalBodyContent.innerHTML = `
    <div class="space-y-4">
      <div>
        <span class="text-[#D9D3CC] font-bold uppercase tracking-wider block mb-1">Input Text:</span>
        <div class="p-3 bg-[#1F1325] rounded-lg border border-[#D9D3CC]/30 text-[#D9D3CC] font-sans text-sm">${escapeHtml(row.input_text || '—')}</div>
      </div>

      ${row.expected_text ? `
        <div>
          <span class="text-[#D9D3CC] font-bold uppercase tracking-wider block mb-1">Expected Ground Truth:</span>
          <div class="p-3 bg-[#1F1325] rounded-lg border border-[#D9D3CC]/30 text-[#FF8C12] font-sans text-sm font-semibold">${escapeHtml(row.expected_text)}</div>
        </div>
      ` : ''}

      <div>
        <span class="text-[#D9D3CC] font-bold uppercase tracking-wider block mb-1">Baseline NLP Output:</span>
        <div class="p-3 bg-[#1F1325] rounded-lg border border-[#D9D3CC]/30 text-[#D9D3CC] font-sans text-sm">${escapeHtml(row.baseline_text || '—')}</div>
        <div class="mt-1 text-[#D9D3CC] font-mono text-[11px] flex gap-4">
          <span>Accuracy: <strong class="text-[#FF8C12]">${row.baseline_accuracy || '—'}</strong></span>
          <span>Precision: <strong class="text-[#D9D3CC]">${row.baseline_precision || '—'}</strong></span>
          <span>Recall: <strong class="text-[#D9D3CC]">${row.baseline_recall || '—'}</strong></span>
          <span>F1: <strong class="text-[#FF8C12]">${row.baseline_f1 || '—'}</strong></span>
        </div>
      </div>

      ${row.llm_text ? `
        <div>
          <span class="text-[#D9D3CC] font-bold uppercase tracking-wider block mb-1">LLM Output:</span>
          <div class="p-3 bg-[#1F1325] rounded-lg border border-[#D9D3CC]/30 text-[#D9D3CC] font-sans text-sm whitespace-pre-wrap">${escapeHtml(row.llm_text)}</div>
          <div class="mt-1 text-[#D9D3CC] font-mono text-[11px] flex gap-4">
            <span>Accuracy: <strong class="text-[#FF8C12]">${row.llm_accuracy || '—'}</strong></span>
            <span>Precision: <strong class="text-[#D9D3CC]">${row.llm_precision || '—'}</strong></span>
            <span>Recall: <strong class="text-[#D9D3CC]">${row.llm_recall || '—'}</strong></span>
            <span>F1: <strong class="text-[#FF8C12]">${row.llm_f1 || '—'}</strong></span>
          </div>
        </div>
      ` : ''}

      <div class="pt-2 border-t border-[#D9D3CC]/25">
        <span class="text-[#D9D3CC] font-bold uppercase tracking-wider block mb-1">Raw Record Fields:</span>
        <div class="max-h-48 overflow-y-auto bg-[#1F1325] rounded p-2.5 text-[11px] font-mono space-y-1 border border-[#D9D3CC]/20">
          ${Object.entries(row).map(([k, v]) => `<div><span class="text-[#C7BFB5] font-semibold">${escapeHtml(k)}:</span> <span class="text-[#D9D3CC]">${escapeHtml(v)}</span></div>`).join('')}
        </div>
      </div>
    </div>
  `;
  dom.caseModal.classList.add('active');
}

// ----------------- Batch Evaluator -----------------

async function startBatchEvaluation() {
  const casesFile = dom.batchCasesFile.value;
  const outputFile = dom.batchOutputFilename.value.trim() || 'batch_eval_results.csv';

  const models = [];
  if (dom.batchCheckT5.checked) models.push('vennify/t5-base-grammar-correction');
  if (dom.batchCheckFlan.checked) models.push('google/flan-t5-small');

  dom.btnStartBatch.disabled = true;
  dom.batchProgressBox.classList.remove('hidden');
  dom.batchProgressBar.style.width = '10%';
  dom.batchProgressPct.textContent = '10%';
  dom.batchProgressStatus.textContent = `Starting evaluation of ${casesFile}...`;
  dom.batchLogOutput.innerHTML = `<div>[${new Date().toLocaleTimeString()}] Initializing benchmark engine...</div>`;

  try {
    const res = await fetch('/api/benchmark/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cases_file: casesFile,
        output_file: outputFile,
        models: models
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Batch evaluation failed');
    }

    const result = await res.json();
    dom.batchProgressBar.style.width = '100%';
    dom.batchProgressPct.textContent = '100%';
    dom.batchProgressStatus.textContent = 'Benchmark Finished!';
    dom.batchLogOutput.innerHTML += `
      <div class="text-[#FF8C12] font-bold">[${new Date().toLocaleTimeString()}] Done! Evaluated ${result.cases_count} cases.</div>
      <div class="text-[#D9D3CC] font-semibold">Saved to: ${result.output_file}</div>
    `;

    showToast(`Batch completed! Saved to ${result.output_file}`, 'success');
    loadCsvFileList(result.output_file);
  } catch (err) {
    dom.batchProgressStatus.textContent = 'Error occurred';
    dom.batchLogOutput.innerHTML += `<div class="text-rose-400">Error: ${err.message}</div>`;
    showToast(err.message, 'error');
  } finally {
    dom.btnStartBatch.disabled = false;
  }
}

// ----------------- Explainability & Sandbox -----------------

async function explainWord(word) {
  if (!word) return;
  try {
    const res = await fetch('/api/explain/word', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word: word })
    });
    const data = await res.json();

    dom.explainVocabStatus.textContent = data.in_vocabulary ? 'In Vocabulary (Valid)' : 'Out of Vocabulary (Misspelled)';
    dom.explainVocabStatus.className = `text-xs font-mono px-2.5 py-0.5 rounded ${data.in_vocabulary ? 'bg-[#D9D3CC]/20 text-[#D9D3CC] border border-[#D9D3CC]/35 font-bold' : 'bg-rose-500/20 text-rose-300 border border-rose-500/30 font-semibold'}`;
    dom.explainBestWord.textContent = data.corrected;
    dom.explainDistance.textContent = `d = ${data.distance}`;
    dom.explainConfidence.textContent = `${(data.confidence * 100).toFixed(0)}% (relative score)`;

    if (data.candidates && data.candidates.length > 0) {
      dom.explainCandidatesList.innerHTML = data.candidates.map(c => `
        <span class="px-2.5 py-1 rounded bg-[#D9D3CC]/12 border border-[#D9D3CC]/30 text-[#D9D3CC] font-mono">
          ${escapeHtml(c.word)} <span class="text-[#FF8C12] text-[10px] font-bold">d=${c.distance}</span> <span class="text-[#D9D3CC]/70 text-[10px]">f=${c.freq}</span>
        </span>
      `).join('');
    } else {
      dom.explainCandidatesList.innerHTML = `<span class="text-[#D9D3CC]/60 italic">No candidates within distance threshold</span>`;
    }
  } catch (err) {
    console.error(err);
  }
}

async function loadVocabularyStats() {
  try {
    const res = await fetch('/api/vocabulary');
    const data = await res.json();
    state.vocabulary = data;
    state.vocabulary.set = new Set(data.top.map(item => item.word.toLowerCase()));

    dom.statVocabSize.textContent = `${data.size} words`;
    dom.statCorpusTokens.textContent = `${data.tokens} tokens`;

    dom.topLexiconChips.innerHTML = data.top.slice(0, 40).map(item => `
      <span class="px-2.5 py-1 rounded-full bg-[#D9D3CC]/10 border border-[#D9D3CC]/25 text-[#D9D3CC] text-xs font-mono hover:bg-[#D9D3CC]/20 transition-colors">
        ${escapeHtml(item.word)} <span class="text-[#FF8C12] text-[10px] font-bold">(${item.freq})</span>
      </span>
    `).join('');

    updateInputStats();
  } catch (err) {
    console.error(err);
  }
}

async function loadSystemInfo() {
  try {
    const res = await fetch('/api/info');
    const info = await res.json();
    state.sysInfo = info;

    document.getElementById('header-device-badge').textContent = `${info.torch_device || 'CPU'}`;
    document.getElementById('sys-device').textContent = info.torch_device || 'CPU';
    document.getElementById('sys-torch').textContent = info.torch_version || '2.x';
    document.getElementById('sys-transformers').textContent = info.transformers_version || '4.40+';
  } catch (err) {
    console.error(err);
  }
}

// ----------------- Utility Functions -----------------

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
