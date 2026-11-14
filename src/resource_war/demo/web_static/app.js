// Resource War demo client. Owner: Rohit Gangil, Reviewer: Atharv Singhal
// All game logic runs in Python (resource_war.demo.trace); this file only draws
// frames, steps through them, and sends your decisions back to the server.
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";
const ACTIONS = ["A", "D", "E"];
const NAME = { A: "Attack", D: "Defend", E: "Expand" };
const ICON = { A: "#i-attack", D: "#i-defend", E: "#i-expand" };
const PLAYER = ["Blue", "Orange"];
const CELL = 100;

const S = {
  meta: null,
  mode: "watch",
  trace: null,
  i: 0,
  timer: null,
  decisions: [],
  advice: true,
  params: {},
  scores: [0, 0],
  units: new Map(),
  lastRendered: null,
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ico = (href, cls = "") => `<svg class="${cls}" aria-hidden="true"><use href="${href}"/></svg>`;
const pct = (x) => `${Math.round(x * 100)}%`;
const fmt = (x) => (Number.isInteger(x) ? String(x) : x.toFixed(2));

// ------------------------------------------------------------------ server
async function api(path, body) {
  showBusy(true);
  try {
    const res = await fetch(path, { method: body ? "POST" : "GET", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);
    showError(null);
    return data;
  } catch (err) {
    showError(err.message.includes("Failed to fetch") ? "The demo server is not running. Start it with: python -m resource_war.demo.web" : err.message);
    throw err;
  } finally {
    showBusy(false);
  }
}

function showBusy(on) { $("busy").hidden = !on; }
function showError(msg) { const e = $("error"); e.hidden = !msg; e.textContent = msg || ""; }

// ------------------------------------------------------------------ board
function buildBoard() {
  const n = S.meta.board_size;
  const b = $("board");
  b.setAttribute("viewBox", `-34 -34 ${n * CELL + 34} ${n * CELL + 34}`);
  let html = `<defs>
    <marker id="ah0" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5L0 10z" fill="#3987e5"/></marker>
    <marker id="ah1" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5L0 10z" fill="#d95926"/></marker>
  </defs>`;
  for (let c = 0; c < n; c++) html += `<text class="coord" x="${c * CELL + 50}" y="-12">${c}</text>`;
  for (let r = 0; r < n; r++) html += `<text class="coord" x="-17" y="${r * CELL + 55}">${r}</text>`;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++)
    html += `<rect class="tile${(r + c) % 2 ? " alt" : ""}" x="${c * CELL + 3}" y="${r * CELL + 3}" width="94" height="94" rx="10"/>`;
  html += `<g id="g-own"></g><g id="g-arrows"></g><g id="g-bases"></g><g id="g-crystals"></g><g id="g-focus"></g><g id="g-units"></g><g id="g-badges"></g>`;
  b.innerHTML = html;
  S.units = new Map();
  S.lastRendered = null;
}

function ensureUnits(units) {
  const layer = $("g-units");
  for (const [uid, p] of units) {
    if (S.units.has(uid)) continue;
    const g = document.createElementNS(SVGNS, "g");
    g.setAttribute("class", `unit p${p}`);
    g.innerHTML = `<circle class="body" r="14"/><use class="emblem" href="${p === 0 ? "#i-chevron" : "#i-diamond"}" x="-8" y="-8" width="16" height="16"/>`;
    layer.appendChild(g);
    S.units.set(uid, g);
  }
}

function slotPositions(n, crowded) {
  const rows = n <= 3 ? [n] : [Math.ceil(n / 2), Math.floor(n / 2)];
  const ys = crowded ? (rows.length === 1 ? [76] : [66, 87]) : rows.length === 1 ? [50] : [38, 64];
  const gap = n <= 3 ? 28 : 24;
  const out = [];
  rows.forEach((count, ri) => {
    for (let k = 0; k < count; k++) out.push([50 + (k - (count - 1) / 2) * gap, ys[ri]]);
  });
  return out;
}

function renderBoard(frame) {
  const b = frame.board;
  const n = S.meta.board_size;
  const bases = [[0, 0], [n - 1, n - 1]];
  const isBase = (r, c) => bases.some(([br, bc]) => br === r && bc === c);
  const resources = new Map(b.resources.map(([r, c, v]) => [`${r},${c}`, v]));

  $("g-own").innerHTML = b.owner
    .map(([r, c, p]) => {
      const x = c * CELL + 3, y = r * CELL + 3;
      const mark = p === 0 ? `<path d="M${x + 6} ${y + 6}h16l-16 16z" fill="#3987e5"/>` : `<path d="M${x + 14} ${y + 5}l7 7-7 7-7-7z" fill="#d95926"/>`;
      return `<rect class="own p${p}" x="${x + 1.5}" y="${y + 1.5}" width="91" height="91" rx="9"/>${isBase(r, c) ? "" : mark}`;
    })
    .join("");

  $("g-bases").innerHTML = bases.map(([r, c], p) =>
    `<use href="#i-castle" x="${c * CELL + 26}" y="${r * CELL + 12}" width="48" height="48" style="color:${p ? "#d95926" : "#3987e5"}"/>`).join("");

  const occupied = new Set(b.units.map(([, , r, c]) => `${r},${c}`));
  $("g-crystals").innerHTML = b.resources
    .map(([r, c, v]) => {
      const s = 13 + v * 1.7;
      const cx = c * CELL + 50, cy = r * CELL + (occupied.has(`${r},${c}`) ? 36 : 48);
      return `<g class="crystal" transform="translate(${cx} ${cy})">
        <path class="gem" d="M0 ${-s}L${s * 0.75} ${-s * 0.18}L0 ${s}L${-s * 0.75} ${-s * 0.18}Z"/>
        <path class="facet" d="M0 ${-s}L0 ${s}L${-s * 0.75} ${-s * 0.18}Z"/>
        <circle class="badge" cx="${s * 0.82}" cy="${s * 0.55}" r="12"/><text class="val" x="${s * 0.82}" y="${s * 0.55 + 1}">${fmt(v)}</text></g>`;
    })
    .join("");

  const moves = frame.highlight?.moves || [];
  $("g-arrows").innerHTML = moves
    .map(([fr, fc, tr, tc, p]) => {
      const off = p === 0 ? -6 : 6;
      const x1 = fc * CELL + 50 + off, y1 = fr * CELL + 50 + off, x2 = tc * CELL + 50 + off, y2 = tr * CELL + 50 + off;
      const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
      return `<line class="arrow p${p}" x1="${x1 + (dx / len) * 22}" y1="${y1 + (dy / len) * 22}" x2="${x1 + dx - (dx / len) * 22}" y2="${y1 + dy - (dy / len) * 22}" marker-end="url(#ah${p})"/>`;
    })
    .join("");

  const cells = (frame.highlight?.cells || []).filter(Boolean);
  const pulse = frame.kind === "encounter" || frame.kind === "pending";
  $("g-focus").innerHTML = cells.map(([r, c]) => `<rect class="focus-ring${pulse ? " pulse" : ""}" x="${c * CELL + 2}" y="${r * CELL + 2}" width="96" height="96" rx="11"/>`).join("");

  // units: grouped per cell, persistent elements so they glide between frames.
  // Only a single step forward or back animates; first draws and jumps snap into place.
  const layer = $("g-units");
  const snap = S.lastRendered == null || Math.abs(S.i - S.lastRendered) !== 1;
  layer.classList.toggle("no-anim", snap);
  ensureUnits(b.units.map(([uid, p]) => [uid, p]));
  const groups = new Map();
  for (const [uid, p, r, c] of b.units) {
    const key = `${r},${c}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push([uid, p, r, c]);
  }
  for (const [key, list] of groups) {
    list.sort((a, z) => a[1] - z[1] || a[0] - z[0]);
    const [r, c] = key.split(",").map(Number);
    const crowded = resources.has(key) || isBase(r, c);
    const pos = slotPositions(list.length, crowded);
    list.forEach(([uid], k) => {
      const g = S.units.get(uid);
      const radius = list.length <= 3 ? 14 : 11;
      g.querySelector("circle").setAttribute("r", radius);
      g.style.transform = `translate(${c * CELL + pos[k][0]}px, ${r * CELL + pos[k][1]}px)`;
    });
  }

  // action badges at a standoff
  const enc = frame.encounter;
  let badges = "";
  if (enc && enc.players && cells[0]) {
    const [r, c] = cells[0];
    enc.players.forEach((pl, p) => {
      const x = c * CELL + (p === 0 ? 20 : 80), y = r * CELL + 16;
      badges += `<g class="act-badge p${p}"><circle cx="${x}" cy="${y}" r="17"/><use href="${ICON[pl.action]}" x="${x - 10}" y="${y - 10}" width="20" height="20"/></g>`;
    });
  }
  $("g-badges").innerHTML = badges;
  if (snap) {
    getComputedStyle(layer.firstElementChild || layer).transform; // apply positions before re-enabling motion
    requestAnimationFrame(() => layer.classList.remove("no-anim"));
  }
  S.lastRendered = S.i;
  $("board").setAttribute("aria-label", `Board at turn ${b.turn}: Blue ${b.scores[0]}, Orange ${b.scores[1]}`);
}

// ------------------------------------------------------------------ scoreboard
function renderScore(frame) {
  const b = frame.board;
  const agents = S.trace.agents;
  const depths = S.trace.depths;
  for (const p of [0, 1]) {
    $(`name${p}`).textContent = agents[p].kind === "human" ? "You" : agents[p].label;
    $(`stats${p}`).textContent = `looks ahead ${depths[p]}; ${fmt(b.collected[p])} in crystals, ${b.territory[p]} tiles`;
    const pts = $(`pts${p}`);
    const val = fmt(b.scores[p]);
    if (pts.textContent !== val) {
      pts.textContent = val;
      const plate = $(`plate${p}`);
      plate.classList.remove("flash");
      void plate.offsetWidth;
      plate.classList.add("flash");
    }
  }
  $("turn").textContent = b.turn;
  $("turnof").textContent = `of ${S.meta.turn_limit}`;
}

// ------------------------------------------------------------------ panel pieces
function bars(probs, side, picked, title) {
  return `<div class="bars"><div class="bars-title">${esc(title)}</div>${ACTIONS.map((a, k) => `
    <div class="bar-row${picked === k ? " picked" : ""}">${ico(ICON[a])}<span>${NAME[a]}</span>
      <div class="track"><div class="fill" style="width:${pct(probs[k])}"></div></div><span class="num">${pct(probs[k])}</span></div>`).join("")}</div>`;
}

function sideCard(pl, p, chosenIdx) {
  const label = pl.kind === "human" ? "You" : pl.label;
  let body = `<p class="why">${esc(pl.why.rule)}</p>`;
  if (pl.why.belief) {
    body += bars(pl.why.belief, p, null, `What it expects ${PLAYER[1 - p]} to do`);
    body += `<div class="bars-title">Expected payoff of each move: ${ACTIONS.map((a, k) => `${NAME[a]} ${fmt(pl.why.values[k])}`).join(", ")}</div>`;
  } else if (pl.kind !== "human") {
    body += bars(pl.why.probs, p, chosenIdx, pl.kind === "pure_heuristic" ? "Its rule" : "Its odds for this standoff");
  }
  const gain = pl.payoff == null ? "" : `<span class="gain">${pl.payoff.startsWith("-") ? "" : "+"}${esc(pl.payoff)}</span>`;
  return `<div class="side p${p}">
    <div class="side-head"><div class="token p${p}">${ico(p === 0 ? "#i-chevron" : "#i-diamond")}</div><div><strong>${PLAYER[p]}: ${esc(label)}</strong><span>${pl.kind === "human" ? "you decide" : `looks ahead ${pl.depth} in battles`}</span></div></div>
    <div class="choice"><span class="ico">${ico(ICON[pl.action])}</span>${NAME[pl.action]}${gain}</div>
    ${body}</div>`;
}

function matrixHtml(m, chosen, rowLabel = "Blue", colLabel = "Orange") {
  let h = `<div class="matrix" role="table" aria-label="Payoff matrix: ${rowLabel} rows, ${colLabel} columns"><div class="corner">${esc(rowLabel)} ↓<br>${esc(colLabel)} →</div>`;
  h += ACTIONS.map((a) => `<div class="h">${ico(ICON[a])}${NAME[a]}</div>`).join("");
  ACTIONS.forEach((a, i) => {
    h += `<div class="h row">${ico(ICON[a])}${NAME[a]}</div>`;
    ACTIONS.forEach((b, j) => {
      const [u, v] = m.cells[i][j];
      const pick = chosen && chosen[0] === i && chosen[1] === j;
      h += `<div class="cell${pick ? " picked" : ""}"><span class="b">${esc(u)}</span><span class="o">${esc(v)}</span></div>`;
    });
  });
  return h + "</div>";
}

function analysisNotes(a) {
  const items = [a.banner];
  if (a.strictly_dominated.length) items.push(`Never worth playing: ${a.strictly_dominated.join(", ")} is strictly dominated.`);
  if (a.weakly_dominated.length) items.push(`${a.weakly_dominated.join(", ")} is only weakly dominated, so it can still appear in an equilibrium.`);
  items.push(a.equilibrium_text);
  if (a.pareto_text) items.push(a.pareto_text);
  return `<ul class="notes">${items.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>`;
}

function encounterHtml(frame) {
  const e = frame.encounter;
  let h = "";
  if (e.players) h += `<div class="duel">${e.players.map((pl, p) => sideCard(pl, p, e.chosen[p])).join("")}</div>`;
  if (e.matrix) {
    h += `<div><h2 class="section-title">The payoff matrix for this crystal (Blue's payoff, Orange's payoff)</h2>${matrixHtml(e.matrix, e.chosen)}${analysisNotes(e.analysis)}</div>`;
  }
  return h;
}

function fighter(u, side, k, actor, target) {
  const lost = Math.max(0, 3 - u.hp);
  const pips = Array.from({ length: u.hp }, () => "<i></i>").join("") + Array.from({ length: lost }, () => '<i class="lost"></i>').join("");
  return `<div class="fighter${u.hp === 0 ? " out" : ""}${actor ? " actor" : ""}${target ? " hit" : ""}">
      <div class="token p${side}">${ico(side === 0 ? "#i-chevron" : "#i-diamond")}</div>
      ${u.defending ? `<svg class="guard" aria-label="guarding"><use href="#i-defend"/></svg>` : ""}
      <div class="hp" aria-label="${u.hp} HP">${pips}</div><div class="name">${u.hp === 0 ? "knocked out" : `unit ${k + 1}`}</div></div>`;
}

function arenaHtml(bt, frame) {
  const st = bt.state;
  const ply = bt.ply;
  const actor = ply ? [ply.side, ply.move.unit] : null;
  const target = ply && ply.move.action === "STRIKE" ? [1 - ply.side, ply.move.target] : null;
  const cellFor = (side, k, rank) => {
    const u = st.sides[side][k];
    const pos = u.hp === 0 ? (u.pos === "withdrawn" ? "withdrawn" : u.pos) : u.pos;
    if (pos !== rank) return "";
    return fighter(u, side, k, actor && actor[0] === side && actor[1] === k, target && target[0] === side && target[1] === k);
  };
  let grid = `<div class="ranks"><div class="lab">Blue rear</div><div class="lab">Blue front</div><div></div><div class="lab">Orange front</div><div class="lab">Orange rear</div>`;
  const rows = Math.max(st.sides[0].length, st.sides[1].length);
  for (let k = 0; k < rows; k++) {
    grid += `<div class="slot">${cellFor(0, k, "back")}</div><div class="slot front">${cellFor(0, k, "front")}</div><div class="mid">${k === 0 ? "vs" : ""}</div>`;
    grid += `<div class="slot front">${cellFor(1, k, "front")}</div><div class="slot">${cellFor(1, k, "back")}</div>`;
  }
  const withdrawn = [0, 1].flatMap((s) => st.sides[s].map((u, k) => (u.pos === "withdrawn" && u.hp > 0 ? `${PLAYER[s]} unit ${k + 1}` : null)).filter(Boolean));
  grid += `</div>`;
  if (withdrawn.length) grid += `<div class="withdrawn">Withdrawn (safe, but no longer counted): ${withdrawn.join(", ")}</div>`;

  let h = `<div class="arena"><h2 class="section-title">Battle: HP in the fight, Blue ${st.hp[0]} vs Orange ${st.hp[1]}${st.ply != null ? `, move ${st.ply} of 10` : ""}</h2>${grid}`;
  if (ply) {
    const blueView = ply.side === 0 ? ply.value : -ply.value;
    const proven = Math.abs(ply.value) >= 900;
    const pos = proven ? (blueView > 0 ? 100 : 0) : 50 + Math.max(-50, Math.min(50, (blueView / 60) * 50));
    const verdict = proven ? `${blueView > 0 ? "Blue" : "Orange"} has a proven win if both play perfectly` : blueView === 0 ? "Level" : `${blueView > 0 ? "Blue" : "Orange"} is ahead (${blueView > 0 ? "+" : ""}${blueView} for Blue)`;
    h += `<div class="evalbar"><div class="bars-title">${esc(PLAYER[ply.side])}'s search verdict: ${esc(verdict)}</div><div class="track"><span class="marker" style="left:${pos}%"></span></div><div class="cap"><span>Orange winning</span><span>Blue winning</span></div></div>`;
    h += `<div><h2 class="section-title">Moves ${esc(PLAYER[ply.side])} considered (searched ${ply.depth} ahead)</h2><ol class="lines">${ply.ranked
      .slice(0, 6)
      .map((m) => `<li class="${m.chosen ? "played" : ""}"><span class="v">${m.value > 0 ? "+" : ""}${m.value}</span><span>${esc(m.label)}</span><span class="tag">${m.chosen ? "played" : m.value === ply.best_value ? "also best" : ""}</span></li>`)
      .join("")}</ol></div>`;
    if (ply.minimax) {
      const mm = ply.minimax.nodes, ab = ply.alphabeta.nodes;
      h += `<div class="search"><h2 class="section-title">Same move, less work</h2>
        <div class="row"><span>Minimax</span><div class="track"><div class="fill mm" style="width:100%"></div></div></div>
        <div class="row"><span>Alpha-beta</span><div class="track"><div class="fill ab" style="width:${Math.max(1, (ab / mm) * 100)}%"></div></div></div>
        <p>Minimax checked ${mm.toLocaleString()} positions in ${ply.minimax.ms.toFixed(1)} ms; alpha-beta checked ${ab.toLocaleString()} in ${ply.alphabeta.ms.toFixed(1)} ms and pruned ${ply.alphabeta.pruned.toLocaleString()} branches it could prove were irrelevant: ${pct(1 - ab / mm)} less work, identical decision.</p></div>`;
    }
  }
  return h + "</div>";
}

function pendingHtml(frame) {
  const pd = frame.pending;
  const toggle = `<label class="advice-toggle"><input type="checkbox" id="advice" ${S.advice ? "checked" : ""}> Show advice</label>`;
  if (pd.kind === "encounter") {
    const best = Math.max(...pd.ev_vs_equilibrium);
    const buttons = ACTIONS.map((a, k) => `<button type="button" class="act" data-choice="${a}">${ico(ICON[a])}<b>${NAME[a]}</b>
        ${S.advice ? `<small>earns ${fmt(pd.ev_vs_equilibrium[k])} against equilibrium play</small>` : ""}<kbd>${k + 1}</kbd></button>`).join("");
    let h = `<div class="yourmove"><h2>Your move</h2><p class="why">A ${fmt(pd.value)}-crystal at (${pd.cell.join(", ")}). Your opponent, ${esc(pd.opponent.label)}, is choosing right now. ${esc(pd.opponent.blurb)}</p>
      <div class="actions">${buttons}</div>${toggle}</div>`;
    if (S.advice) h += `<div><h2 class="section-title">The payoff matrix (your payoff first)</h2>${matrixHtml(pd.matrix, null, "You", "Them")}${analysisNotes(pd.analysis)}<p class="why">${esc(pd.ev_text)}${best - Math.min(...pd.ev_vs_equilibrium) < 1e-9 ? " They are all equal: that is exactly what an equilibrium does to you." : ""}</p></div>`;
    return h;
  }
  const buttons = pd.moves.map((m, k) => `<button type="button" data-move="${m.index}"><kbd>${k + 1}</kbd><span>${esc(m.label)}</span>
      <span>${S.advice ? `${m.value > 0 ? "+" : ""}${m.value}${m.value === pd.best_value ? ' <span class="best">best</span>' : ""}` : ""}</span></button>`).join("");
  return `${arenaHtml(frame.battle, frame)}<div class="yourmove"><h2>Your battle move</h2><p class="why">Pick one move for one of your units. ${S.advice ? "Numbers are what your own search (minimax) thinks each move is worth: positive is good for you." : ""}</p><div class="moves">${buttons}</div>${toggle}</div>`;
}

function renderPanel(frame) {
  $("title").textContent = frame.title;
  const n = S.trace.frames.length;
  $("where").textContent = `Step ${S.i + 1} of ${n}, turn ${frame.turn}`;
  $("story").innerHTML = (frame.text || []).map((t) => `<li>${esc(t)}</li>`).join("");
  let h = "";
  const live = frame.kind === "pending" && S.i === n - 1;
  if (live) h += pendingHtml(frame);
  else if (frame.kind === "pending") h += `<p class="why">This is where the game is waiting for you. Step forward to the last frame to choose.</p>`;
  else if (frame.encounter) h += encounterHtml(frame);
  if (frame.battle && !live) h += arenaHtml(frame.battle, frame);
  if (frame.kind === "end") h += `<p class="why">Use the timeline to revisit any standoff or battle, or start a new match with another seed.</p>`;
  if (S.mode === "play" && frame.decision_index != null && !live) {
    h += `<p><button type="button" class="btn" id="rewind" data-index="${frame.decision_index}">${ico("#i-undo")}Go back and choose differently here</button></p>`;
  }
  $("detail").innerHTML = h;
  wirePanel();
}

function wirePanel() {
  document.querySelectorAll("[data-choice]").forEach((b) => b.addEventListener("click", () => decide({ kind: "encounter", choice: b.dataset.choice })));
  document.querySelectorAll("[data-move]").forEach((b) => b.addEventListener("click", () => decide({ kind: "battle", choice: Number(b.dataset.move) })));
  const adv = $("advice");
  if (adv) adv.addEventListener("change", () => { S.advice = adv.checked; render(); });
  const rw = $("rewind");
  if (rw) rw.addEventListener("click", () => rewind(Number(rw.dataset.index)));
}

// ------------------------------------------------------------------ timeline
function renderMarks() {
  const frames = S.trace.frames;
  const n = frames.length;
  $("marks").innerHTML = frames
    .map((f, k) => {
      const left = n > 1 ? (k / (n - 1)) * 100 : 0;
      let cls = null;
      if (f.decision_index != null || f.kind === "pending") cls = "you";
      else if (f.kind === "encounter") cls = "enc";
      else if (f.kind === "battle_ply") cls = "bat";
      else if (f.kind === "end") cls = "end";
      return cls ? `<span class="mark ${cls}" style="left:${left}%"></span>` : "";
    })
    .join("");
  $("scrub").max = n - 1;
}

function render() {
  const frame = S.trace.frames[S.i];
  renderScore(frame);
  renderBoard(frame);
  renderPanel(frame);
  $("scrub").value = S.i;
  const n = S.trace.frames.length;
  $("where-now").textContent = `Step ${S.i + 1} of ${n}`;
  $("b-prev").disabled = $("b-first").disabled = S.i === 0;
  $("b-next").disabled = $("b-last").disabled = S.i >= n - 1;
  $("b-undo").hidden = S.mode !== "play" || S.decisions.length === 0;
  saveHash();
}

function go(i) {
  if (!S.trace) return;
  S.i = Math.max(0, Math.min(S.trace.frames.length - 1, i));
  render();
}

function setPlaying(on) {
  clearInterval(S.timer);
  S.timer = null;
  $("b-play").innerHTML = ico(on ? "#i-pause" : "#i-play");
  $("b-play").setAttribute("aria-label", on ? "Pause" : "Play");
  if (!on) return;
  S.timer = setInterval(() => {
    if (!S.trace || S.i >= S.trace.frames.length - 1) return setPlaying(false);
    go(S.i + 1);
  }, Number($("speed").value));
}

function nextStandoff() {
  const frames = S.trace.frames;
  for (let k = S.i + 1; k < frames.length; k++) if (frames[k].kind === "encounter" || frames[k].kind === "pending") return go(k);
  go(frames.length - 1);
}

// ------------------------------------------------------------------ modes
async function startWatch() {
  const p = {
    agents: [$("w-a").value, $("w-b").value],
    depths: [Number($("w-da").value), Number($("w-db").value)],
    seed: Number($("w-seed").value),
  };
  setPlaying(false);
  const trace = await api("/api/watch", p);
  S.params = p;
  load(trace, 0);
}

async function startPlay(decisions = [], jumpFrom = null) {
  const p = {
    opponent: $("p-opp").value,
    depths: [Number($("p-dy").value), Number($("p-do").value)],
    seed: Number($("p-seed").value),
    decisions,
  };
  setPlaying(false);
  const trace = await api("/api/play", p);
  S.params = p;
  S.decisions = decisions;
  const last = trace.frames.length - 1;
  load(trace, jumpFrom == null ? last : Math.min(jumpFrom, last));
  if (jumpFrom != null && jumpFrom < last) setPlaying(true); // replay what happened after your choice
}

function load(trace, i) {
  S.trace = trace;
  buildBoard();
  renderMarks();
  S.i = i;
  render();
}

async function decide(d) {
  const from = S.i;
  try { await startPlay([...S.decisions, d], from); } catch { /* error shown */ }
}

async function rewind(index) {
  try { await startPlay(S.decisions.slice(0, index)); } catch { /* error shown */ }
}

function setMode(mode) {
  S.mode = mode;
  document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
  $("setup-watch").hidden = mode !== "watch";
  $("setup-play").hidden = mode !== "play";
  $("stage").hidden = mode === "explore";
  $("timeline").hidden = mode === "explore";
  $("explore").hidden = mode !== "explore";
  setPlaying(false);
  if (mode === "watch") startWatch().catch(() => {});
  else if (mode === "play") startPlay(S.decisionsFromHash || []).catch(() => {});
  else explore();
  S.decisionsFromHash = null;
}

// ------------------------------------------------------------------ explore
let exploreTimer = null;
async function explore() {
  const v = (id) => Number($(id).value);
  for (const [id, o] of [["x-v", "o-v"], ["x-c", "o-c"], ["x-d", "o-d"], ["x-pr", "o-pr"], ["x-pc", "o-pc"]]) $(o).textContent = fmt(v(id));
  const body = { value: v("x-v"), attack_cost: v("x-c"), defend_cost: v("x-d"), p_row: v("x-pr"), p_col: v("x-pc") };
  saveHash(body);
  let data;
  try { data = await api("/api/explore", body); } catch { return; }
  const twoC = data.two_c;
  $("x-strip").style.setProperty("--peace", `${Math.min(100, (twoC / 12) * 100)}%`);
  $("x-2c").style.setProperty("--peace", `${Math.min(100, (twoC / 12) * 100)}%`);
  $("x-pin").style.left = `${(body.value / 12) * 100}%`;
  $("x-2c").textContent = `2c = ${fmt(twoC)}`;
  const regime = { below: "Below 2c: attacking never pays, so both sides expand and share.", "exactly at": "Exactly at 2c: the edge case. Attack is only weakly dominated and there are infinitely many equilibria.", above: "Above 2c: rock-paper-scissors. Defend beats Attack, Expand beats Defend, Attack beats Expand." };
  $("x-banner").textContent = regime[data.regime];
  const a = data.analysis;
  $("x-out").innerHTML = `<h2 class="section-title">Payoff matrix (Blue's payoff, Orange's payoff)</h2>${matrixHtml(data.matrix, null)}
    <div class="duel" style="margin-top:14px"><div class="side p0">${bars(a.equilibrium_row, 0, null, "Blue's equilibrium odds")}</div><div class="side p1">${bars(a.equilibrium_col, 1, null, "Orange's equilibrium odds")}</div></div>
    ${analysisNotes(a)}
    <p class="why">Try moving the battle-win sliders: when Blue gets better at battles, it is Orange's odds that change, not Blue's. In a mixed equilibrium each side's odds are set to keep the other side indifferent.</p>`;
}

// ------------------------------------------------------------------ url state
function saveHash(explore) {
  let h;
  if (S.mode === "explore") {
    const e = explore || {};
    h = `#explore&v=${e.value}&c=${e.attack_cost}&d=${e.defend_cost}&pr=${e.p_row}&pc=${e.p_col}`;
  } else if (S.mode === "watch" && S.params.agents) {
    h = `#watch&a=${S.params.agents[0]}&b=${S.params.agents[1]}&da=${S.params.depths[0]}&db=${S.params.depths[1]}&seed=${S.params.seed}&step=${S.i}`;
  } else if (S.mode === "play" && S.params.opponent) {
    const d = S.decisions.map((x) => (x.kind === "encounter" ? x.choice : `b${x.choice}`)).join(".");
    h = `#play&opp=${S.params.opponent}&dy=${S.params.depths[0]}&do=${S.params.depths[1]}&seed=${S.params.seed}&d=${d}&step=${S.i}`;
  }
  if (h && h !== location.hash) history.replaceState(null, "", h);
}

function readHash() {
  const parts = location.hash.replace(/^#/, "").split("&");
  const mode = ["watch", "play", "explore"].includes(parts[0]) ? parts[0] : "watch";
  const q = Object.fromEntries(parts.slice(1).map((s) => s.split("=")));
  const set = (id, v) => { if (v != null && v !== "" && v !== "undefined") $(id).value = v; };
  if (mode === "watch") { set("w-a", q.a); set("w-b", q.b); set("w-da", q.da); set("w-db", q.db); set("w-seed", q.seed); }
  if (mode === "play") {
    set("p-opp", q.opp); set("p-dy", q.dy); set("p-do", q.do); set("p-seed", q.seed);
    S.decisionsFromHash = (q.d || "").split(".").filter(Boolean).map((s) => (s[0] === "b" ? { kind: "battle", choice: Number(s.slice(1)) } : { kind: "encounter", choice: s }));
  }
  if (mode === "explore") { set("x-v", q.v); set("x-c", q.c); set("x-d", q.d); set("x-pr", q.pr); set("x-pc", q.pc); }
  S.stepFromHash = q.step != null ? Number(q.step) : null;
  return mode;
}

// ------------------------------------------------------------------ setup
function fillSelect(id, options, value) {
  $(id).innerHTML = options.map(([v, label]) => `<option value="${esc(v)}"${String(v) === String(value) ? " selected" : ""}>${esc(label)}</option>`).join("");
}

async function init() {
  try { S.meta = await api("/api/meta"); } catch { $("title").textContent = "Cannot reach the demo server"; return; }
  const agents = S.meta.agents.map((a) => [a.kind, a.label]);
  const depths = S.meta.depths.map((d) => [d, `${d} move${d > 1 ? "s" : ""}`]);
  const dw = S.meta.defaults.watch, dp = S.meta.defaults.play;
  fillSelect("w-a", agents, dw.agents[0]); fillSelect("w-b", agents, dw.agents[1]);
  fillSelect("w-da", depths, dw.depths[0]); fillSelect("w-db", depths, dw.depths[1]);
  fillSelect("p-opp", agents, dp.opponent); fillSelect("p-dy", depths, dp.depths[0]); fillSelect("p-do", depths, dp.depths[1]);
  $("x-c").value = S.meta.payoffs.attack_cost; $("x-d").value = S.meta.payoffs.defend_cost;

  document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("setup-watch").addEventListener("submit", (e) => { e.preventDefault(); startWatch().catch(() => {}); });
  $("setup-play").addEventListener("submit", (e) => { e.preventDefault(); startPlay([]).catch(() => {}); });
  document.querySelectorAll("[data-dice]").forEach((b) => b.addEventListener("click", () => { $(b.dataset.dice).value = Math.floor(Math.random() * 100000); }));
  $("b-first").addEventListener("click", () => go(0));
  $("b-prev").addEventListener("click", () => go(S.i - 1));
  $("b-next").addEventListener("click", () => go(S.i + 1));
  $("b-last").addEventListener("click", () => go(Infinity));
  $("b-play").addEventListener("click", () => setPlaying(!S.timer));
  $("b-standoff").addEventListener("click", nextStandoff);
  $("b-undo").addEventListener("click", () => rewind(S.decisions.length - 1));
  $("scrub").addEventListener("input", (e) => { setPlaying(false); go(Number(e.target.value)); });
  $("speed").addEventListener("change", () => { if (S.timer) setPlaying(true); });
  $("how").addEventListener("click", () => $("rules").showModal());
  ["x-v", "x-c", "x-d", "x-pr", "x-pc"].forEach((id) => $(id).addEventListener("input", () => { clearTimeout(exploreTimer); exploreTimer = setTimeout(explore, 120); }));

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea") && e.target.type !== "range") return;
    if (S.mode === "explore" || !S.trace) return;
    const frame = S.trace.frames[S.i];
    const live = frame.kind === "pending" && S.i === S.trace.frames.length - 1;
    if (e.key === "ArrowRight") { setPlaying(false); go(S.i + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { setPlaying(false); go(S.i - 1); e.preventDefault(); }
    else if (e.key === " ") { setPlaying(!S.timer); e.preventDefault(); }
    else if (e.key === "Home") go(0);
    else if (e.key === "End") go(Infinity);
    else if (e.key.toLowerCase() === "n") nextStandoff();
    else if (e.key.toLowerCase() === "u" && S.mode === "play" && S.decisions.length) rewind(S.decisions.length - 1);
    else if (live && /^[1-9]$/.test(e.key)) {
      const k = Number(e.key) - 1;
      if (frame.pending.kind === "encounter" && k < 3) decide({ kind: "encounter", choice: ACTIONS[k] });
      if (frame.pending.kind === "battle" && k < frame.pending.moves.length) decide({ kind: "battle", choice: frame.pending.moves[k].index });
    }
  });

  const mode = readHash();
  const step = S.stepFromHash;
  setMode(mode);
  if (step != null && mode !== "explore") {
    const wait = setInterval(() => { if (S.trace) { clearInterval(wait); go(step); } }, 50);
    setTimeout(() => clearInterval(wait), 15000);
  }
}

init();
