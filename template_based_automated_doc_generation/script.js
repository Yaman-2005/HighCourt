// ── Library guard — show clear error if CDN scripts failed ──
(function checkLibs() {
    const missing = [];
    if (typeof PizZip === 'undefined') missing.push('PizZip');
    if (typeof mammoth === 'undefined') missing.push('mammoth');
    if (typeof saveAs === 'undefined') missing.push('FileSaver');
    if (missing.length) {
        document.body.innerHTML = `
      <div style="max-width:600px;margin:80px auto;font-family:Georgia,serif;padding:32px;
        border:2px solid #c0392b;border-radius:4px;background:#fff8f8;color:#c0392b">
        <h2 style="margin:0 0 12px;font-size:20px">⚠ Required libraries failed to load</h2>
        <p style="color:#333;font-size:15px;line-height:1.6">
          The following scripts could not be fetched from the CDN:<br>
          <strong>${missing.join(', ')}</strong>
        </p>
        <p style="color:#333;font-size:14px;line-height:1.6;margin-top:12px">
          <strong>This is a standalone HTML file — do not use npm with it.</strong><br>
          Just open the <code>.html</code> file directly in your browser while connected to the internet.
          If you are behind a firewall, download the scripts manually (see below) and place them
          next to the HTML file, then change the <code>&lt;script src&gt;</code> paths to local paths.
        </p>
        <ul style="color:#333;font-size:13px;margin-top:10px;line-height:2">
          <li><a href="https://unpkg.com/pizzip@3.1.4/dist/pizzip.min.js">pizzip.min.js</a></li>
          <li><a href="https://unpkg.com/mammoth@1.6.0/mammoth.browser.min.js">mammoth.browser.min.js</a></li>
          <li><a href="https://unpkg.com/file-saver@2.0.5/dist/FileSaver.min.js">FileSaver.min.js</a></li>
        </ul>
      </div>`;
        throw new Error('Aborted: missing libraries — ' + missing.join(', '));
    }
})();

const builtInTemplates = [
    {
        id: "lease",
        name: "Lease Agreement",
        file: "templates/lease-agreement.docx",
        tags: ["property", "rent"]
    },
    {
        id: "internship",
        name: "Internship Agreement",
        file: "templates/internship-agreement.docx",
        tags: ["education", "training"]
    },
    {
        id: "petition",
        name: "Petition",
        file: "templates/sample-document.docx",
        tags: ["legal", "case"]
    }
];
// ── State ──
let docxBytes = null;   // original template ArrayBuffer
let placeholders = [];     // ['CASE_NUMBER', ...]
let fieldValues = {};
let filledDocxBytes = null;
let docxFileName = '';

let saved = [];
try { saved = JSON.parse(localStorage.getItem('hc_docx_templates') || '[]'); } catch (e) { saved = []; }

// ── Helpers ──
function show(id, type, msg) {
    const el = document.getElementById(id);
    el.className = `status ${type}`;
    el.innerHTML = msg;
}
function hide(id) { document.getElementById(id).className = 'status'; }
function unlock(id) {
    document.getElementById(id).classList.remove('locked');
    document.getElementById(id).classList.add('unlocked');
}
function ab2b64(buf) {
    let b = ''; const u = new Uint8Array(buf);
    for (let i = 0; i < u.byteLength; i++) b += String.fromCharCode(u[i]);
    return btoa(b);
}
function b642ab(b64) {
    const b = atob(b64), buf = new ArrayBuffer(b.length), v = new Uint8Array(buf);
    for (let i = 0; i < b.length; i++) v[i] = b.charCodeAt(i);
    return buf;
}
const DATE_PATTERNS = [
    'date', 'dob', 'birth', 'day',
    'start', 'end', 'from', 'to',
    'issued', 'expiry', 'valid',
    'hearing', 'filing', 'signed',
    'agreement', 'lease', 'rent'
];

function isDateField(ph) {
    const key = ph.toLowerCase();
    let score = 0;

    DATE_PATTERNS.forEach(p => {
        if (key.includes(p)) score++;
    });

    return score >= 1; // tweak threshold
}
function slugify(s) { return (s || 'document').replace(/[^a-zA-Z0-9]/g, '_'); }
function today() { return new Date().toISOString().slice(0, 10); }

// ── Detect brace style ──
function detectBraceStyle(plain) {
    if (/\{\{[A-Za-z]/.test(plain)) return 'double';
    if (/\{[A-Za-z]/.test(plain)) return 'single';
    return 'double';
}

// ── Extract placeholders from DOCX XML — supports {x} and {{X}} ──
async function extractPlaceholders(arrayBuffer) {
    const zip = new PizZip(arrayBuffer);
    const xmlFiles = [
        'word/document.xml',
        'word/header1.xml', 'word/header2.xml', 'word/header3.xml',
        'word/footer1.xml', 'word/footer2.xml', 'word/footer3.xml',
    ];
    let allText = '';
    for (const f of xmlFiles) {
        try { allText += zip.file(f)?.asText() || ''; } catch (e) { /* file absent */ }
    }
    // Strip XML tags — remove tags entirely so run-split placeholders merge
    const plain = allText.replace(/<[^>]+>/g, '');
    window._braceStyle = detectBraceStyle(plain);

    const found = new Set();
    let m;

    if (window._braceStyle === 'double') {
        const re = /\{\{([A-Za-z][A-Za-z0-9_ ]*?)\}\}/g;
        while ((m = re.exec(plain)) !== null)
            found.add(m[1].trim().replace(/\s+/g, '_'));
    } else {
        // Single brace {camelCase} — skip loop control tags {#x} {/x} {^x}
        const re = /\{([A-Za-z][A-Za-z0-9_]*)\}/g;
        while ((m = re.exec(plain)) !== null) {
            if (m[1].length >= 2) found.add(m[1].trim());
        }
    }

    return [...found].sort();
}
async function loadBuiltIn(index) {
    const template = builtInTemplates[index];

    try {
        const res = await fetch(template.file);

        if (!res.ok) {
            throw new Error(`HTTP ${res.status} - could not load file`);
        }

        const blob = await res.blob();
        docxBytes = await blob.arrayBuffer();

        docxFileName = template.name + ".docx";

        placeholders = await extractPlaceholders(docxBytes);

        // UI updates
        document.querySelectorAll('.tmpl-card').forEach(c => c.classList.remove('active'));

        document.getElementById('dropZone').style.display = 'none';
        document.getElementById('filePill').style.display = 'flex';
        document.getElementById('pillName').textContent = template.name;

        document.getElementById('s1btn').disabled = false;

        show('s1msg', 'ok',
            `Template "${template.name}" loaded — ${placeholders.length} placeholder(s) found.`
        );

    } catch (e) {
        console.error(e);
        show('s1msg', 'err', `Failed to load template "${template.name}"`);
    }
}
// ── Saved templates ──
function renderSaved() {
    //   const lib = document.getElementById('tmplLib');
    //   const orDiv = document.getElementById('orDivider');
    //   if (!saved.length) { lib.innerHTML = ''; orDiv.style.display = 'none'; return; }
    //   orDiv.style.display = '';
    //   lib.innerHTML = saved.map((t, i) => `
    //     <div class="tmpl-card" onclick="loadSaved(${i})" id="tc_${i}">
    //       <div style="font-size:20px;flex-shrink:0">📋</div>
    //       <div style="flex:1;min-width:0">
    //         <div class="tmpl-name">${t.name}</div>
    //         <div class="tmpl-file">${t.fileName}</div>
    //         <div class="tmpl-tags">
    //           ${t.placeholders.slice(0,5).map(p=>`<span class="mini-tag">${t.braceStyle === 'single' ? '{'+p+'}' : '{{'+p+'}}'}</span>`).join('')}
    //           ${t.placeholders.length>5?`<span class="mini-tag">+${t.placeholders.length-5} more</span>`:''}
    //         </div>
    //       </div>
    //       <button class="tmpl-del" onclick="event.stopPropagation();delSaved(${i})" title="Remove">✕</button>
    //     </div>`).join('');
    const lib = document.getElementById('tmplLib');
    const orDiv = document.getElementById('orDivider');

    let html = '';

    // ── Built-in templates ──
    html += builtInTemplates.map((t, i) => `
    <div class="tmpl-card" onclick="loadBuiltIn(${i})">
      <div style="font-size:20px;flex-shrink:0">📄</div>
      <div style="flex:1;min-width:0">
        <div class="tmpl-name">${t.name}</div>
        <div class="tmpl-tags">
          ${t.tags.map(tag => `<span class="mini-tag">${tag}</span>`).join('')}
        </div>
      </div>
    </div>
  `).join('');

    // ── Saved templates ──
    if (saved.length) {
        // show divider only if saved exists
        if (orDiv) orDiv.style.display = '';

        html += `<div class="divider-or">Saved Templates</div>`;

        html += saved.map((t, i) => `
      <div class="tmpl-card" onclick="loadSaved(${i})" id="tc_${i}">
        <div style="font-size:20px;flex-shrink:0">📋</div>
        <div style="flex:1;min-width:0">
          <div class="tmpl-name">${t.name}</div>
          <div class="tmpl-file">${t.fileName}</div>
          <div class="tmpl-tags">
            ${t.placeholders.slice(0, 5).map(p =>
            `<span class="mini-tag">${t.braceStyle === 'single' ? '{' + p + '}' : '{{' + p + '}}'
            }</span>`
        ).join('')}
            ${t.placeholders.length > 5
                ? `<span class="mini-tag">+${t.placeholders.length - 5} more</span>`
                : ''}
          </div>
        </div>
        <button class="tmpl-del"
          onclick="event.stopPropagation();delSaved(${i})"
          title="Remove">✕</button>
      </div>
    `).join('');
    } else {
        if (orDiv) orDiv.style.display = 'none';
    }

    lib.innerHTML = html;
}

// function renderAllTemplates() {
//   const lib = document.getElementById('tmplLib');

//   let html = '';

//   // Built-in templates
//   html += builtInTemplates.map((t, i) => `
//     <div class="tmpl-card" onclick="loadBuiltIn(${i})">
//       <div style="font-size:20px">📄</div>
//       <div style="flex:1">
//         <div class="tmpl-name">${t.name}</div>
//         <div class="tmpl-tags">
//           ${t.tags.map(tag => `<span class="mini-tag">${tag}</span>`).join('')}
//         </div>
//       </div>
//     </div>
//   `).join('');

//   // Divider
//   if (saved.length) {
//     html += `<div class="divider-or">Saved Templates</div>`;
//   }

//   // Saved templates (your existing logic)
//   html += saved.map((t, i) => `
//     <div class="tmpl-card" onclick="loadSaved(${i})">
//       <div style="font-size:20px">📋</div>
//       <div style="flex:1">
//         <div class="tmpl-name">${t.name}</div>
//         <div class="tmpl-file">${t.fileName}</div>
//       </div>
//       <button class="tmpl-del" onclick="event.stopPropagation();delSaved(${i})">✕</button>
//     </div>
//   `).join('');

//   lib.innerHTML = html;
// }

async function loadSaved(i) {
    const t = saved[i];
    docxBytes = b642ab(t.data);
    placeholders = t.placeholders;
    docxFileName = t.fileName;
    document.querySelectorAll('.tmpl-card').forEach(c => c.classList.remove('active'));
    document.getElementById(`tc_${i}`).classList.add('active');
    document.getElementById('dropZone').style.display = 'none';
    document.getElementById('filePill').style.display = 'flex';
    document.getElementById('pillName').textContent = `${t.name}  (${t.fileName})`;
    document.getElementById('s1btn').disabled = false;
    show('s1msg', 'inf', `Template "${t.name}" selected — ${placeholders.length} placeholder(s) ready.`);
}

function delSaved(i) {
    if (!confirm(`Remove template "${saved[i].name}"?`)) return;
    saved.splice(i, 1);
    localStorage.setItem('hc_docx_templates', JSON.stringify(saved));
    renderSaved();
}

// ── File input ──
const fileIn = document.getElementById('fileIn');
const dropZone = document.getElementById('dropZone');

fileIn.addEventListener('change', async e => {
    if (e.target.files[0]) await handleFile(e.target.files[0]);
});
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', async e => {
    e.preventDefault(); dropZone.classList.remove('over');
    const f = e.dataTransfer.files[0];
    if (f && (f.name.endsWith('.docx') || f.type.includes('wordprocessingml'))) await handleFile(f);
    else show('s1msg', 'err', 'Please drop a .docx (Word document) file.');
});

async function handleFile(file) {
    docxFileName = file.name;
    docxBytes = await file.arrayBuffer();
    try {
        placeholders = await extractPlaceholders(docxBytes);
    } catch (e) {
        show('s1msg', 'err', 'Could not read DOCX: ' + e.message); return;
    }
    document.getElementById('dropZone').style.display = 'none';
    document.getElementById('filePill').style.display = 'flex';
    document.getElementById('pillName').textContent = file.name;
    document.getElementById('saveRow').style.display = 'flex';
    document.getElementById('saveName').value = file.name.replace('.docx', '');
    document.getElementById('s1btn').disabled = false;
    if (placeholders.length > 0)
        show('s1msg', 'ok', `✓ Found ${placeholders.length} placeholder(s): ${placeholders.map(p => fmtTag(p)).join(', ')}`);
    else
        show('s1msg', 'err', 'No placeholders found. Ensure your DOCX contains tags like {camelCase} or {{UPPER_CASE}}.');
}

function resetFile() {
    docxBytes = null; placeholders = []; docxFileName = '';
    document.getElementById('dropZone').style.display = '';
    document.getElementById('filePill').style.display = 'none';
    document.getElementById('saveRow').style.display = 'none';
    document.getElementById('s1btn').disabled = true;
    fileIn.value = '';
    document.querySelectorAll('.tmpl-card').forEach(c => c.classList.remove('active'));
    hide('s1msg');
}

// ── Step 1 → Step 2 ──
async function doStep1() {
    if (!docxBytes || !placeholders.length) {
        show('s1msg', 'err', 'No valid DOCX with placeholders loaded.'); return;
    }
    const ck = document.getElementById('saveCk');
    const nm = document.getElementById('saveName').value.trim();
    if (ck.checked && nm) {
        const idx = saved.findIndex(t => t.name === nm);
        const entry = { name: nm, fileName: docxFileName, placeholders, braceStyle: window._braceStyle || 'double', data: ab2b64(docxBytes) };
        if (idx >= 0) saved[idx] = entry; else saved.push(entry);
        localStorage.setItem('hc_docx_templates', JSON.stringify(saved));
        renderSaved();
    }
    buildForm();
    unlock('s2');
    document.getElementById('s2').scrollIntoView({ behavior: 'smooth' });
}

// ── Build form ──
const LONG = ['ORDER', 'JUDGMENT', 'BODY', 'FACTS', 'SUBMISSIONS', 'REASONS', 'RELIEF', 'PRAYER', 'SCHEDULE', 'DESCRIPTION', 'DETAILS', 'REMARKS', 'NARRATION', 'CONTENT', 'TEXT', 'HISTORY'];
const HINTS = {
    CASE_NUMBER: 'e.g. C.S. COM. No. 123 of 2025',
    DATE: 'e.g. 13th January 2025',
    JUDGE_NAME: "e.g. Hon'ble Justice A.K. Sharma",
    PLAINTIFF: 'Full name of plaintiff',
    DEFENDANT: 'Full name of defendant',
    COURT: 'e.g. High Court at Calcutta',
    YEAR: 'e.g. 2025',
    ADVOCATE: 'Name of advocate',
    AMOUNT: 'e.g. Rs. 61,60,585/-',
    ADDRESS: 'Full address',
};

function fmtTag(p) {
    return window._braceStyle === 'single' ? `{${p}}` : `{{${p}}}`;
}
function humanLabel(p) {
    // camelCase → "Camel Case", SNAKE_CASE → "Snake Case"
    return p.replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2')
        .replace(/\b\w/g, c => c.toUpperCase());
}

function buildForm() {
    const grid = document.getElementById('fGrid');
    grid.innerHTML = '';
    document.getElementById('tagCloud').innerHTML = placeholders.map(p => `<span class="tag">${fmtTag(p)}</span>`).join('');
    document.getElementById('tagArea').style.display = 'block';

    placeholders.forEach(ph => {
        const label = humanLabel(ph);
        const isLong = LONG.some(l => ph.toUpperCase().includes(l));
        const isFull = isLong || ph.toUpperCase().includes('ADDRESS') || ph.toUpperCase().includes('PARTIES');
        const div = document.createElement('div');
        div.className = `fg${isFull ? ' full' : ''}`;

        const lbl = document.createElement('label');
        lbl.htmlFor = `f_${ph}`;
        const [ob, cb] = window._braceStyle === 'single' ? ['{', '}'] : ['{{', '}}'];
        lbl.innerHTML = `<span class="br">${ob}</span>${label}<span class="br">${cb}</span> <span class="req">*</span>`;

        let inp;
        // if (isLong) {
        //   inp = document.createElement('textarea');
        //   inp.className = 'ft'; inp.rows = 5;
        // } else {
        //   inp = document.createElement('input');
        //   inp.type = 'text'; inp.className = 'fi';
        // }
        if (isLong) {
            inp = document.createElement('textarea');
            inp.className = 'ft';
            inp.rows = 5;

        } else if (isDateField(ph)) {
            // 📅 Auto calendar for detected date fields
            inp = document.createElement('input');
            inp.type = 'date';
            inp.className = 'fi';

        } else {
            inp = document.createElement('input');
            inp.type = 'text';
            inp.className = 'fi';
        }
        inp.id = `f_${ph}`;
        // Match hints by uppercase version for camelCase keys
        const hintKey = ph.replace(/([a-z])([A-Z])/g, '$1_$2').replace(/_/g, '_').toUpperCase();
        inp.placeholder = HINTS[hintKey] || HINTS[ph] || `Enter ${humanLabel(ph).toLowerCase()}`;
        inp.addEventListener('input', checkAll);

        div.appendChild(lbl); div.appendChild(inp);
        grid.appendChild(div);
    });
    grid.className = placeholders.length <= 2 ? 'fields-grid single' : 'fields-grid';
    checkAll();
}

function checkAll() {
    const ok = placeholders.every(ph => {
        const el = document.getElementById(`f_${ph}`);
        return el && el.value.trim();
    });
    document.getElementById('s2btn').disabled = !ok;
}

function clearFields() {
    placeholders.forEach(ph => { const el = document.getElementById(`f_${ph}`); if (el) el.value = ''; });
    checkAll();
}

function formatDateIndian(value) {
    if (!value) return '';
    const [y, m, d] = value.split('-');
    return `${d}/${m}/${y}`;
}

// ── Step 2 → Step 3 ──
async function doStep2() {
    fieldValues = {};
    placeholders.forEach(ph => {
        const el = document.getElementById(`f_${ph}`);
        // fieldValues[ph] = el ? el.value.trim() : '';
        if (el) {
            if (el.type === 'date') {
                fieldValues[ph] = formatDateIndian(el.value);
            } else {
                fieldValues[ph] = el.value.trim();
            }
        } else {
            fieldValues[ph] = '';
        }
    });

    // Field values table
    document.getElementById('prevList').innerHTML = placeholders.map(ph => `
    <li>
      <span class="prev-k">${fmtTag(ph)}</span>
      <span class="prev-v ${fieldValues[ph] ? '' : 'mt'}">${fieldValues[ph] || '(empty)'}</span>
    </li>`).join('');

    show('s3msg', 'inf', 'Generating filled document…');

    try {
        // Build filled DOCX
        filledDocxBytes = await buildFilledDocx(docxBytes, fieldValues);
        window.currentDocxBlob = new Blob([filledDocxBytes], {
            type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        });
        // Preview via mammoth
        const arr = new Uint8Array(filledDocxBytes);
        const result = await mammoth.convertToHtml({ arrayBuffer: arr.buffer });
        document.getElementById('docxPreview').innerHTML = result.value || '<p style="color:#bbb;font-style:italic">No content to preview.</p>';

        hide('s3msg');
        unlock('s3');
        unlock('s4');
        document.getElementById('s3').scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
        show('s3msg', 'err', 'Error generating document: ' + e.message);
        console.error(e);
    }
}

// ── Build filled DOCX using PizZip (raw XML replacement) ──
async function buildFilledDocx(srcBuf, values) {
    const zip = new PizZip(srcBuf);

    // Target XML files in the DOCX package
    const targets = [
        'word/document.xml',
        'word/header1.xml', 'word/header2.xml', 'word/header3.xml',
        'word/footer1.xml', 'word/footer2.xml', 'word/footer3.xml',
    ];

    for (const target of targets) {
        const file = zip.file(target);
        if (!file) continue;

        let xml = file.asText();

        // DOCX often splits placeholder text across multiple XML runs.
        // Step 1: collapse run-split placeholders by merging adjacent <w:t> contents.
        // We do a safe approach: strip tags from runs inside a paragraph to reassemble,
        // replace, then re-wrap in a single run.
        xml = mergeAndReplace(xml, values);

        zip.file(target, xml);
    }

    return zip.generate({
        type: 'arraybuffer',
        mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        compression: 'DEFLATE',
    });
}

// ── Merge split runs and replace placeholders ──
function mergeAndReplace(xml, values) {
    const single = (window._braceStyle === 'single');

    // First pass: direct XML replacement for non-split placeholders
    for (const [key, val] of Object.entries(values)) {
        const tag = single ? `{${key}}` : `{{${key}}}`;
        const tagAlt = single ? `{${key.replace(/_/g, ' ')}}` : `{{${key.replace(/_/g, ' ')}}}`;
        const escaped = escapeXml(val);
        xml = xml.split(tag).join(escaped);
        xml = xml.split(tagAlt).join(escaped);
    }

    // For single-brace templates: strip loop control tags {#x}...{/x} and {^x}
    if (single) {
        xml = xml.replace(/\{[#^\/][A-Za-z][A-Za-z0-9_]*\}/g, '');
    }

    // Second pass: handle run-split placeholders
    xml = fixSplitPlaceholders(xml, values);

    return xml;
}

function escapeXml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&apos;');
}

function fixSplitPlaceholders(xml, values) {
    // Within each <w:p>...</w:p>, collect all text, check for split placeholders,
    // and if found, replace in the joined text, then rewrite runs.
    return xml.replace(/<w:p[ >][\s\S]*?<\/w:p>/g, para => {
        // Extract text content of all runs (ignoring XML tags)
        const textContent = [];
        const runRegex = /<w:t[^>]*>([\s\S]*?)<\/w:t>/g;
        let m;
        while ((m = runRegex.exec(para)) !== null) {
            textContent.push(m[1]);
        }
        const fullText = textContent.join('');

        const single = (window._braceStyle === 'single');
        let needsRebuild = false;
        for (const key of Object.keys(values)) {
            const tag = single ? `{${key}}` : `{{${key}}}`;
            if (fullText.includes(tag)) { needsRebuild = true; break; }
        }
        if (!needsRebuild) return para;

        let replaced = fullText;
        for (const [key, val] of Object.entries(values)) {
            const tag = single ? `{${key}}` : `{{${key}}}`;
            replaced = replaced.split(tag).join(val);
        }
        // Strip leftover loop tags in reassembled text
        if (single) replaced = replaced.replace(/\{[#^\/][A-Za-z][A-Za-z0-9_]*\}/g, '');

        // Rewrite: replace the text content of the first <w:t> with the full replaced text,
        // remove subsequent runs' text, preserving paragraph properties and first run's properties.
        let runIdx = 0;
        const newPara = para.replace(/<w:r[ >][\s\S]*?<\/w:r>/g, run => {
            if (runIdx === 0) {
                runIdx++;
                // Replace this run's w:t content with full replaced text
                return run.replace(/<w:t[^>]*>[\s\S]*?<\/w:t>/, `<w:t xml:space="preserve">${escapeXml(replaced)}</w:t>`);
            } else {
                runIdx++;
                // Remove text from subsequent runs (they were consumed into first run)
                return run.replace(/<w:t[^>]*>[\s\S]*?<\/w:t>/, '<w:t></w:t>');
            }
        });
        return newPara;
    });
}

// ── Download DOCX ──
function downloadDOCX() {
    if (!filledDocxBytes) { show('s4msg', 'err', 'Generate the preview first.'); return; }
    const blob = new Blob([filledDocxBytes], {
        type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    });
    const caseNum = fieldValues['CASE_NUMBER'] || fieldValues[Object.keys(fieldValues)[0]] || 'document';
    const name = `${slugify(caseNum)}_${today()}.docx`;
    saveAs(blob, name);
    show('s4msg', 'ok', `✓ Word document downloaded as "${name}"`);
}


// ── Init ──
renderSaved();