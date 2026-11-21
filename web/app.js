/**
 * LexiFix AI — Modern Interactive Application Logic
 * Proportional Colour Palette:
 * 1. Pebble Silk  (#D9D3CC) - 60% (Dominant Canvas & Surfaces)
 * 2. Eclipse Plum (#2B1C33) - 20% (Structural Framing & Deep Typography)
 * 3. Amber Flare  (#FF8C12) - 20% (Radiant Energy, CTAs & Accents)
 */

// Application State
const state = {
  currentView: 'studio',
  theme: localStorage.getItem('lexifix_theme') || 'light',
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
  
  // Explain default sample word
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
  dom.explainSoundex = document.getElementById('explain-soundex');
  dom.explainConfidence = document.getElementById('explain-confidence');
  dom.explainCandidatesList = document.getElementById('explain-candidates-list');
  dom.statVocabSize = document.getElementById('stat-vocab-size');
  dom.statCorpusTokens = document.getElementById('stat-corpus-tokens');
  dom.topLexiconChips = document.getElementById('top-lexicon-chips');

  // Modals & Header
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
}

function bindEvents() {
  // Navigation tabs
  dom.navButtons.forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.nav));
  });

  // Theme toggle
  dom.btnThemeToggle.addEventListener('click', toggleTheme);
  if (dom.settingsThemeSelect) {
    dom.settingsThemeSelect.addEventListener('change', (e) => applyTheme(e.target.value));
  }

  // Real-time input analysis
  dom.inputText.addEventListener('input', updateInputStats);
  dom.btnClearInput.addEventListener('click', () => {
    dom.inputText.value = '';
    updateInputStats();
    dom.inputText.focus();
  });

  // Expected Ground Truth toggle
  dom.btnToggleExpected.addEventListener('click', () => {
    dom.expectedContainer.classList.remove('hidden');
    dom.expectedText.focus();
  });
  dom.btnRemoveExpected.addEventListener('click', () => {
    dom.expectedContainer.classList.add('hidden');
    dom.expectedText.value = '';
  });

  // Slider edit distance
  dom.sliderMaxDistance.addEventListener('input', (e) => {
    dom.labelMaxDistance.textContent = `${e.target.value} edit${e.target.value > 1 ? 's' : ''}`;
  });

  // Quick sample chips
  dom.sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      dom.inputText.value = chip.dataset.sample;
      updateInputStats();
      triggerCorrection();
    });
  });

  // Custom model drawer
  dom.btnToggleCustom.addEventListener('click', () => {
    dom.customModelDrawer.classList.toggle('hidden');
    if (!dom.customModelDrawer.classList.contains('hidden')) {
      dom.inputCustomModel.focus();
    }
  });

  // Main Action: Correct & Compare
  dom.btnCorrect.addEventListener('click', triggerCorrection);

  // Dynamic View Mode Toggling (Diff vs Clean vs Explain)
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.view-mode-btn');
    if (btn) {
      const parent = btn.parentElement;
      const target = btn.dataset.target;
      const mode = btn.dataset.mode;

      parent.querySelectorAll('.view-mode-btn').forEach(b => {
        b.classList.remove('active', 'bg-[#FF8C12]', 'text-[#2B1C33]', 'font-bold');
        b.classList.add('text-[#4E3758]');
      });
      btn.classList.add('active', 'bg-[#FF8C12]', 'text-[#2B1C33]', 'font-bold');
      btn.classList.remove('text-[#4E3758]');
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
    btn.classList.toggle('bg-[#FF8C12]/20', isActive);
    btn.classList.toggle('border-[#FF8C12]/60', isActive);
    btn.classList.toggle('text-[#2B1C33]', isActive);
    btn.classList.toggle('font-bold', isActive);
    btn.classList.toggle('text-[#4E3758]', !isActive);
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

// ----------------- Real-Time Token Analysis -----------------

function updateInputStats() {
  const text = dom.inputText.value;
  const words = (text.match(/[A-Za-z]+(?:'[A-Za-z]+)?/g) || []);
  const chars = text.length;

  dom.statWords.textContent = `${words.length} word${words.length === 1 ? '' : 's'}`;
  dom.statChars.textContent = `${chars} character${chars === 1 ? '' : 's'}`;

  if (state.vocabulary.set && state.vocabulary.set.size > 0) {
    const unknownWords = words.filter(w => !state.vocabulary.set.has(w.toLowerCase()));
    const uniqueUnknown = [...new Set(unknownWords.map(w => w.toLowerCase()))];
    const count = uniqueUnknown.length;

    dom.detectedBadge.textContent = `${count} error${count === 1 ? '' : 's'} detected`;
    if (count > 0) {
      dom.detectedBadge.className = 'px-2 py-0.5 rounded-full bg-[#FF8C12]/20 text-[#B85B00] text-[10px] font-mono border border-[#FF8C12]/40 font-bold';
      dom.liveDetectedPreview.textContent = `(${uniqueUnknown.slice(0, 3).join(', ')}${count > 3 ? '...' : ''})`;
    } else {
      dom.detectedBadge.className = 'px-2 py-0.5 rounded-full bg-[#2B1C33]/10 text-[#2B1C33] text-[10px] font-mono border border-[#2B1C33]/20 font-medium';
      dom.liveDetectedPreview.textContent = '';
    }
  }
}

// ----------------- Studio Correction Core -----------------

async function triggerCorrection() {
  const text = dom.inputText.value.trim();
  if (!text) {
    showToast('Please enter some text to correct.', 'error');
    return;
  }

  // Gather selected models
  const selectedModels = [];
  dom.modelChecks.forEach(chk => {
    if (chk.checked) selectedModels.push(chk.value);
  });

  const customModel = dom.inputCustomModel.value.trim();
  if (customModel && !selectedModels.includes(customModel)) {
    selectedModels.push(customModel);
  }

  const maxDistance = parseInt(dom.sliderMaxDistance.value, 10);
  const expectedText = dom.expectedText.value.trim();

  // Set processing state
  state.isProcessing = true;
  dom.btnCorrect.disabled = true;
  dom.correctSpinner.classList.remove('hidden');
  dom.correctBtnText.textContent = 'Processing...';

  // Render Skeleton Placeholders for LLMs
  renderLoadingSkeleton(selectedModels);

  try {
    const response = await fetch('/api/correct', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        expected_text: expectedText,
        max_distance: maxDistance,
        models: selectedModels
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    state.lastResult = data;
    renderResults(data);
    dom.exportBar.classList.remove('hidden');

  } catch (err) {
    showToast(`Execution error: ${err.message}`, 'error');
  } finally {
    state.isProcessing = false;
    dom.btnCorrect.disabled = false;
    dom.correctSpinner.classList.add('hidden');
    dom.correctBtnText.textContent = '✦ Correct & Compare';
  }
}

function renderLoadingSkeleton(selectedModels) {
  dom.baselineOutputDiff.innerHTML = `
    <div class="space-y-2 w-full">
      <div class="skeleton h-4 w-3/4"></div>
      <div class="skeleton h-4 w-1/2"></div>
    </div>
  `;
  dom.baselineOutputClean.innerHTML = '';
  dom.baselineOutputExplain.innerHTML = '';

  dom.llmCardsGrid.innerHTML = selectedModels.map(model => `
    <div class="surface-card p-5 space-y-4 border border-[#2B1C33]/20 border-l-4 border-l-[#FF8C12]">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full bg-[#FF8C12] animate-pulse"></span>
          <h3 class="font-bold text-sm tracking-wide text-[#2B1C33]">${escapeHtml(model)}</h3>
          <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-[#FF8C12]/20 text-[#B85B00] font-bold border border-[#FF8C12]/40">Processing...</span>
        </div>
      </div>
      <div class="p-4 bg-[#FFFFFF] rounded-xl border border-[#2B1C33]/20 space-y-2 shadow-inner">
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

  // 2. Baseline Explainability Chips
  if (data.corrections && data.corrections.length > 0) {
    dom.baselineOutputExplain.innerHTML = `
      <div class="space-y-2">
        <span class="text-xs font-bold text-[#2B1C33]">Multi-Signal Token Diagnostic Breakdown:</span>
        <div class="flex flex-wrap gap-2">
          ${data.corrections.map(c => `
            <div class="correction-pill" title="Levenshtein: ${c.distance}, Damerau: ${c.damerau_distance || c.distance}, Confidence: ${(c.confidence * 100).toFixed(0)}%">
              <span class="text-[#D92546] line-through font-semibold">${escapeHtml(c.original)}</span>
              <span class="text-[#4E3758] font-bold">→</span>
              <span class="text-[#B85B00] font-bold">${escapeHtml(c.corrected)}</span>
              <span class="text-[10px] text-[#2B1C33] px-1.5 py-0.5 rounded bg-[#FAF7F4] border border-[#2B1C33]/20 font-bold">d=${c.distance}</span>
              ${c.phonetic_match ? `<span class="text-[9px] text-[#B85B00] bg-[#FF8C12]/20 px-1 py-0.5 rounded border border-[#FF8C12]/40 font-bold">Soundex Match</span>` : ''}
              <span class="text-[10px] text-[#4E3758] font-mono font-bold">${(c.confidence * 100).toFixed(0)}%</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  } else {
    dom.baselineOutputExplain.innerHTML = `<span class="text-xs text-[#4E3758] font-medium">No spelling modifications detected in vocabulary.</span>`;
  }

  // 3. Baseline Metrics with CER and WER support
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

  // 4. Render LLM Output Cards
  if (data.llm_results && data.llm_results.length > 0) {
    dom.llmCardsGrid.innerHTML = data.llm_results.map((llm, idx) => {
      const isOk = llm.status === 'ok';
      const cardId = `llm-card-${idx}`;
      return `
        <div id="${cardId}" class="surface-card p-5 space-y-4 border border-[#2B1C33]/20 border-l-4 border-l-[#FF8C12]">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div class="flex items-center gap-2.5">
              <span class="w-3 h-3 rounded-full ${isOk ? 'bg-[#FF8C12] shadow shadow-[#FF8C12]/50' : 'bg-[#D92546]'}"></span>
              <h3 class="font-bold text-sm tracking-wide text-[#2B1C33]">${escapeHtml(llm.model)}</h3>
              <span class="text-[10px] font-mono px-2 py-0.5 rounded ${isOk ? 'bg-[#2B1C33]/10 text-[#2B1C33] border border-[#2B1C33]/20 font-bold' : 'bg-rose-500/20 text-rose-700 font-bold'}">
                ${isOk ? 'Transformer' : 'Error'}
              </span>
              ${isOk ? `<span class="text-[10px] font-mono text-[#4E3758] bg-[#FAF7F4] px-2 py-0.5 rounded border border-[#2B1C33]/20 font-semibold">${llm.latency_ms || 0}ms</span>` : ''}
            </div>

            ${isOk ? `
              <div class="flex items-center gap-2">
                <div class="bg-[#FAF7F4] p-0.5 rounded-lg border border-[#2B1C33]/20 flex text-xs">
                  <button class="view-mode-btn active px-2.5 py-1 rounded-md text-[#2B1C33] bg-[#FF8C12] font-bold" data-target="llm-${idx}" data-mode="diff">Visual Diff</button>
                  <button class="view-mode-btn px-2.5 py-1 rounded-md text-[#4E3758] hover:text-[#2B1C33]" data-target="llm-${idx}" data-mode="clean">Clean Text</button>
                </div>

                <button class="btn-copy silk-btn p-1.5 rounded-lg text-[#2B1C33] hover:text-[#FF8C12] transition-colors" data-copy-target="llm-${idx}" title="Copy output">
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                </button>
              </div>
            ` : ''}
          </div>

          <div class="p-4 bg-[#FFFFFF] rounded-xl border border-[#2B1C33]/20 min-h-[64px] shadow-inner">
            ${isOk ? `
              <div id="llm-${idx}-output-diff" class="diff-container text-[#2B1C33]">
                ${generateTokenDiffHtml(original, llm.text)}
              </div>
              <div id="llm-${idx}-output-clean" class="hidden text-[#2B1C33] text-sm font-sans leading-relaxed">${escapeHtml(llm.text)}</div>
            ` : `
              <div class="text-[#D92546] text-xs font-mono p-2 bg-rose-500/10 rounded border border-rose-500/30">
                ${escapeHtml(llm.error || 'Failed to generate output from model')}
              </div>
            `}
          </div>

          ${(isOk && data.expected_text) ? `
            <div class="pt-2 border-t border-[#2B1C33]/15 flex flex-wrap items-center justify-between text-xs text-[#2B1C33] font-mono">
              <div class="flex items-center gap-4">
                <span>Accuracy: <strong class="text-[#B85B00]">${(llm.accuracy * 100).toFixed(1)}%</strong></span>
                <span>Precision: <strong class="text-[#2B1C33]">${llm.precision.toFixed(2)}</strong></span>
                <span>Recall: <strong class="text-[#2B1C33]">${llm.recall.toFixed(2)}</strong></span>
                <span>F1 Score: <strong class="text-[#B85B00]">${llm.f1.toFixed(2)}</strong></span>
                ${llm.cer !== undefined ? `<span class="text-[#4E3758]">CER: <strong>${(llm.cer * 100).toFixed(1)}%</strong></span>` : ''}
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
  let text = '';
  if (target === 'baseline') {
    text = dom.baselineOutputClean.textContent;
  } else {
    const cleanEl = document.getElementById(`${target}-output-clean`);
    if (cleanEl) text = cleanEl.textContent;
  }

  if (text) {
    navigator.clipboard.writeText(text).then(() => {
      showToast('Copied to clipboard!', 'success');
    }).catch(() => {
      showToast('Failed to copy text', 'error');
    });
  }
}

// ----------------- Visual Diff Algorithm -----------------

function generateTokenDiffHtml(source, target) {
  if (!source || !target) return escapeHtml(target || '');

  const srcTokens = tokenizeWords(source);
  const tgtTokens = tokenizeWords(target);

  // Dynamic programming alignment (Levenshtein on word tokens)
  const dp = Array(srcTokens.length + 1).fill(null).map(() => Array(tgtTokens.length + 1).fill(0));

  for (let i = 0; i <= srcTokens.length; i++) dp[i][0] = i;
  for (let j = 0; j <= tgtTokens.length; j++) dp[0][j] = j;

  for (let i = 1; i <= srcTokens.length; i++) {
    for (let j = 1; j <= tgtTokens.length; j++) {
      if (srcTokens[i - 1].toLowerCase() === tgtTokens[j - 1].toLowerCase()) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,     // delete
          dp[i][j - 1] + 1,     // insert
          dp[i - 1][j - 1] + 1  // replace
        );
      }
    }
  }

  // Backtrack to extract diff sequence
  let i = srcTokens.length;
  let j = tgtTokens.length;
  const ops = [];

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && srcTokens[i - 1].toLowerCase() === tgtTokens[j - 1].toLowerCase()) {
      ops.push({ type: 'equal', text: tgtTokens[j - 1] });
      i--;
      j--;
    } else if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) {
      ops.push({ type: 'replace', old: srcTokens[i - 1], new: tgtTokens[j - 1] });
      i--;
      j--;
    } else if (j > 0 && dp[i][j] === dp[i][j - 1] + 1) {
      ops.push({ type: 'insert', text: tgtTokens[j - 1] });
      j--;
    } else if (i > 0 && dp[i][j] === dp[i - 1][j] + 1) {
      ops.push({ type: 'delete', text: srcTokens[i - 1] });
      i--;
    }
  }

  ops.reverse();

  // Format into visual pills
  return ops.map(op => {
    if (op.type === 'equal') {
      return `<span class="diff-token">${escapeHtml(op.text)}</span>`;
    } else if (op.type === 'replace') {
      return `<span class="diff-token"><span class="diff-del">${escapeHtml(op.old)}</span><span class="diff-ins">${escapeHtml(op.new)}</span></span>`;
    } else if (op.type === 'insert') {
      return `<span class="diff-token"><span class="diff-ins">${escapeHtml(op.text)}</span></span>`;
    } else if (op.type === 'delete') {
      return `<span class="diff-token"><span class="diff-del">${escapeHtml(op.text)}</span></span>`;
    }
  }).join(' ');
}

function tokenizeWords(str) {
  const matches = str.match(/([A-Za-z]+(?:'[A-Za-z]+)?|[^\s\w]+|\s+)/g);
  return matches ? matches.filter(t => t.trim().length > 0) : [];
}

// ----------------- Save Run to CSV -----------------

async function saveRunToCsv() {
  if (!state.lastResult) return;
  const res = state.lastResult;

  const row = {
    input_text: res.input_text,
    expected_text: res.expected_text || '',
    baseline_text: res.baseline_text,
    baseline_accuracy: (res.baseline_accuracy || 0).toFixed(4),
    baseline_precision: (res.baseline_precision || 0).toFixed(4),
    baseline_recall: (res.baseline_recall || 0).toFixed(4),
    baseline_f1: (res.baseline_f1 || 0).toFixed(4)
  };

  if (res.llm_results && res.llm_results.length > 0) {
    const firstLlm = res.llm_results[0];
    row.llm_text = firstLlm.text;
    row.llm_accuracy = (firstLlm.accuracy || 0).toFixed(4);
    row.llm_precision = (firstLlm.precision || 0).toFixed(4);
    row.llm_recall = (firstLlm.recall || 0).toFixed(4);
    row.llm_f1 = (firstLlm.f1 || 0).toFixed(4);
    row.llm_status = firstLlm.status;
  }

  try {
    const response = await fetch('/api/csv/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename: 'my_results.csv',
        rows: [row]
      })
    });

    const data = await response.json();
    if (data.status === 'ok') {
      showToast(`Appended row to my_results.csv (${data.total_rows} total rows)`, 'success');
      loadCsvFileList('my_results.csv');
    } else {
      showToast(data.error || 'Failed to save', 'error');
    }
  } catch (err) {
    showToast(`Error saving CSV: ${err.message}`, 'error');
  }
}

// ----------------- CSV Explorer -----------------

async function loadCsvFileList(selectFileName = '') {
  try {
    const res = await fetch('/api/csv/list');
    const files = await res.json();
    state.csvFiles = files;

    dom.csvFileSelect.innerHTML = files.map(f => `
      <option value="${f.name}">${f.name} (${f.rows} rows, ${f.size_kb} KB)</option>
    `).join('');

    const targetFile = selectFileName || (files.find(f => f.name === 'my_cases.csv') ? 'my_cases.csv' : (files[0] ? files[0].name : ''));
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
  dom.benchmarkTbody.innerHTML = `<tr><td class="text-center py-8 text-[#4E3758]" colspan="6">Loading ${filename}...</td></tr>`;

  try {
    const res = await fetch(`/api/csv/read?file=${encodeURIComponent(filename)}`);
    const data = await res.json();
    state.csvData = data;
    state.csvFilteredRows = [...data.rows];

    updateKpis(data.rows);
    renderCsvTable();
  } catch (err) {
    showToast(`Error loading CSV: ${err.message}`, 'error');
  }
}

function updateKpis(rows) {
  dom.kpiTotalCases.textContent = rows.length;

  const baselineF1s = rows.map(r => parseFloat(r.baseline_f1)).filter(n => !isNaN(n));
  const meanBaseF1 = baselineF1s.length ? (baselineF1s.reduce((a, b) => a + b, 0) / baselineF1s.length) : null;
  dom.kpiBaselineF1.textContent = meanBaseF1 !== null ? meanBaseF1.toFixed(3) : '—';

  const llmF1s = rows.map(r => parseFloat(r.llm_f1)).filter(n => !isNaN(n));
  const meanLlmF1 = llmF1s.length ? (llmF1s.reduce((a, b) => a + b, 0) / llmF1s.length) : null;
  dom.kpiLlmF1.textContent = meanLlmF1 !== null ? meanLlmF1.toFixed(3) : '—';

  const baseAccs = rows.map(r => parseFloat(r.baseline_accuracy)).filter(n => !isNaN(n));
  const meanBaseAcc = baseAccs.length ? (baseAccs.reduce((a, b) => a + b, 0) / baseAccs.length) : null;
  dom.kpiAccuracyComp.textContent = meanBaseAcc !== null ? `${(meanBaseAcc * 100).toFixed(1)}%` : '—';
}

function renderCsvTable() {
  const headers = state.csvData.headers;
  const rows = state.csvFilteredRows;

  dom.csvRowCounter.textContent = `Showing ${rows.length} of ${state.csvData.rows.length} rows`;

  if (!headers || headers.length === 0) {
    dom.benchmarkThead.innerHTML = '<tr><th>No Data</th></tr>';
    dom.benchmarkTbody.innerHTML = '<tr><td class="text-center py-8 text-[#4E3758]">Empty dataset</td></tr>';
    return;
  }

  // Render Table Header with Sort Triggers
  dom.benchmarkThead.innerHTML = `
    <tr>
      <th class="w-12 text-center">#</th>
      ${headers.map(h => `
        <th onclick="sortCsvBy('${h}')" title="Sort by ${h}">
          <div class="flex items-center gap-1.5">
            <span>${escapeHtml(h)}</span>
            ${state.csvSortCol === h ? `<span class="text-[#FF8C12]">${state.csvSortAsc ? '▲' : '▼'}</span>` : ''}
          </div>
        </th>
      `).join('')}
    </tr>
  `;

  // Render Table Rows
  if (rows.length === 0) {
    dom.benchmarkTbody.innerHTML = `<tr><td colspan="${headers.length + 1}" class="text-center py-8 text-[#4E3758]">No matching records found.</td></tr>`;
    return;
  }

  dom.benchmarkTbody.innerHTML = rows.map((row, idx) => `
    <tr onclick="inspectCsvRow(state.csvFilteredRows[${idx}])" title="Click to view full inspection">
      <td class="text-center font-mono text-[11px] text-[#4E3758]">${idx + 1}</td>
      ${headers.map(h => {
        const val = row[h] || '';
        const isHigh = h.includes('f1') || h.includes('accuracy');
        const numVal = parseFloat(val);
        const highlightClass = isHigh && !isNaN(numVal) && numVal >= 0.99 ? 'text-[#B85B00] font-bold' : '';
        return `<td class="${highlightClass}">${escapeHtml(val)}</td>`;
      }).join('')}
    </tr>
  `).join('');
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

function sortCsvBy(col) {
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
        <span class="text-[#2B1C33] font-bold uppercase tracking-wider block mb-1">Input Text:</span>
        <div class="p-3 bg-[#FFFFFF] rounded-lg border border-[#2B1C33]/20 text-[#2B1C33] font-sans text-sm shadow-inner">${escapeHtml(row.input_text || '—')}</div>
      </div>

      ${row.expected_text ? `
        <div>
          <span class="text-[#2B1C33] font-bold uppercase tracking-wider block mb-1">Expected Ground Truth:</span>
          <div class="p-3 bg-[#FFFFFF] rounded-lg border border-[#2B1C33]/20 text-[#B85B00] font-sans text-sm font-bold shadow-inner">${escapeHtml(row.expected_text)}</div>
        </div>
      ` : ''}

      <div>
        <span class="text-[#2B1C33] font-bold uppercase tracking-wider block mb-1">Baseline NLP Output:</span>
        <div class="p-3 bg-[#FFFFFF] rounded-lg border border-[#2B1C33]/20 text-[#2B1C33] font-sans text-sm shadow-inner">${escapeHtml(row.baseline_text || '—')}</div>
        <div class="mt-1 text-[#2B1C33] font-mono text-[11px] flex gap-4">
          <span>Accuracy: <strong class="text-[#B85B00]">${row.baseline_accuracy || '—'}</strong></span>
          <span>Precision: <strong class="text-[#2B1C33]">${row.baseline_precision || '—'}</strong></span>
          <span>Recall: <strong class="text-[#2B1C33]">${row.baseline_recall || '—'}</strong></span>
          <span>F1: <strong class="text-[#B85B00]">${row.baseline_f1 || '—'}</strong></span>
        </div>
      </div>

      ${row.llm_text ? `
        <div>
          <span class="text-[#2B1C33] font-bold uppercase tracking-wider block mb-1">LLM Output:</span>
          <div class="p-3 bg-[#FFFFFF] rounded-lg border border-[#2B1C33]/20 text-[#2B1C33] font-sans text-sm whitespace-pre-wrap shadow-inner">${escapeHtml(row.llm_text)}</div>
          <div class="mt-1 text-[#2B1C33] font-mono text-[11px] flex gap-4">
            <span>Accuracy: <strong class="text-[#B85B00]">${row.llm_accuracy || '—'}</strong></span>
            <span>Precision: <strong class="text-[#2B1C33]">${row.llm_precision || '—'}</strong></span>
            <span>Recall: <strong class="text-[#2B1C33]">${row.llm_recall || '—'}</strong></span>
            <span>F1: <strong class="text-[#B85B00]">${row.llm_f1 || '—'}</strong></span>
          </div>
        </div>
      ` : ''}

      <div class="pt-2 border-t border-[#2B1C33]/20">
        <span class="text-[#2B1C33] font-bold uppercase tracking-wider block mb-1">Raw Record Fields:</span>
        <div class="max-h-48 overflow-y-auto bg-[#FFFFFF] rounded p-2.5 text-[11px] font-mono space-y-1 border border-[#2B1C33]/20 shadow-inner">
          ${Object.entries(row).map(([k, v]) => `<div><span class="text-[#4E3758] font-bold">${escapeHtml(k)}:</span> <span class="text-[#2B1C33]">${escapeHtml(v)}</span></div>`).join('')}
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
  dom.batchLogOutput.innerHTML = `<div>[${new Date().toLocaleTimeString()}] Initializing batch evaluation...</div>`;

  try {
    dom.batchProgressBar.style.width = '40%';
    dom.batchProgressPct.textContent = '40%';
    dom.batchProgressStatus.textContent = 'Processing dataset cases through models...';
    dom.batchLogOutput.innerHTML += `<div>[${new Date().toLocaleTimeString()}] Running baseline and LLMs on ${casesFile}...</div>`;

    const res = await fetch('/api/benchmark/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cases_file: casesFile,
        output_file: outputFile,
        models: models
      })
    });

    const data = await res.json();
    if (data.status === 'ok') {
      dom.batchProgressBar.style.width = '100%';
      dom.batchProgressPct.textContent = '100%';
      dom.batchProgressStatus.textContent = `Complete! Processed ${data.cases_processed} cases.`;
      dom.batchLogOutput.innerHTML += `<div class="text-[#FF8C12] font-bold">[${new Date().toLocaleTimeString()}] Successfully saved results to ${data.output_file}</div>`;
      showToast(`Batch completed: ${data.cases_processed} cases evaluated`, 'success');
      loadCsvFileList(data.output_file);
    } else {
      throw new Error(data.error || 'Evaluation failed');
    }
  } catch (err) {
    dom.batchProgressStatus.textContent = 'Error occurred';
    dom.batchLogOutput.innerHTML += `<div class="text-[#D92546]">Error: ${err.message}</div>`;
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
    dom.explainVocabStatus.className = `text-xs font-mono px-2.5 py-0.5 rounded ${data.in_vocabulary ? 'bg-emerald-500/15 text-emerald-800 border border-emerald-500/30 font-bold' : 'bg-rose-500/15 text-rose-800 border border-rose-500/30 font-bold'}`;
    dom.explainBestWord.textContent = data.corrected;
    dom.explainDistance.textContent = `Lev: ${data.distance} | Dam: ${data.damerau_distance !== undefined ? data.damerau_distance : data.distance}`;
    if (dom.explainSoundex) {
      dom.explainSoundex.textContent = `Soundex: ${data.soundex || '—'}`;
    }
    dom.explainConfidence.textContent = `${(data.confidence * 100).toFixed(0)}% (relative score)`;

    if (data.candidates && data.candidates.length > 0) {
      dom.explainCandidatesList.innerHTML = data.candidates.map(c => `
        <span class="px-2.5 py-1 rounded bg-[#FAF7F4] border border-[#2B1C33]/20 text-[#2B1C33] font-mono shadow-sm flex items-center gap-1.5">
          <span class="font-bold">${escapeHtml(c.word)}</span>
          <span class="text-[#B85B00] text-[10px] font-bold">d=${c.distance}</span>
          ${c.phonetic_match ? `<span class="bg-[#FF8C12]/20 text-[#B85B00] px-1 py-0.2 rounded text-[9px] font-bold border border-[#FF8C12]/30">Soundex</span>` : ''}
          <span class="text-[#4E3758] text-[10px]">f=${c.freq}</span>
        </span>
      `).join('');
    } else {
      dom.explainCandidatesList.innerHTML = `<span class="text-[#4E3758] italic">No candidates within distance threshold</span>`;
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
      <span class="px-2.5 py-1 rounded-full bg-[#FAF7F4] border border-[#2B1C33]/20 text-[#2B1C33] text-xs font-mono hover:border-[#FF8C12] transition-colors shadow-sm">
        ${escapeHtml(item.word)} <span class="text-[#B85B00] text-[10px] font-bold">(${item.freq})</span>
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

// ----------------- Toast Notifications -----------------

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 350);
  }, 3500);
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
