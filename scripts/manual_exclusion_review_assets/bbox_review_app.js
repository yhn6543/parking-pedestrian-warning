const $ = id => document.getElementById(id);
let rows = [];
let i = 0;
let item = null;
let boxes = [];
let rev = 0;
let visible = localStorage.bboxVisible !== '0';
let sel = -1;
let add = false;
let drag = null;
let undo = [];
let redo = [];
let overlay = false;
const im = new Image();

async function api(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!data.ok) throw new Error(data.error);
  return data;
}

function canvasPoint(event) {
  const rect = $('canvas').getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * im.naturalWidth / rect.width,
    y: (event.clientY - rect.top) * im.naturalHeight / rect.height,
  };
}

function snapshot() {
  undo.push(JSON.stringify(boxes));
  redo = [];
}

function isDirty() {
  return undo.length > 0;
}

function draw() {
  const canvas = $('canvas');
  const context = canvas.getContext('2d');
  canvas.width = im.naturalWidth;
  canvas.height = im.naturalHeight;
  canvas.style.width = '100%';
  context.drawImage(im, 0, 0);
  $('vis').textContent = visible ? 'BBox 표시 중 (V)' : 'BBox 숨김 (V)';
  if (visible) {
    boxes.forEach((box, index) => {
      context.strokeStyle = index === sel ? '#00ffff' : '#00ff55';
      context.lineWidth = index === sel ? 4 : 2;
      context.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
      context.fillStyle = context.strokeStyle;
      context.fillText(index + 1, box.x1, box.y1 - 3);
    });
  }
  $('state').textContent = add
    ? 'Add Box Mode: drag a person area, Esc cancels'
    : `BBox ${visible ? '표시 중' : '숨김'} | Selected: ${sel < 0 ? 'none' : `${sel + 1}/${boxes.length}`}${isDirty() ? ' | unsaved' : ''}`;
}

async function load() {
  const candidateData = await api('/api/candidates');
  const currentId = rows[i]?.normalized_id;
  rows = candidateData.rows;
  if (!rows.length) {
    $('pos').textContent = '0/0';
    return;
  }
  const nextId = currentId || rows[0].normalized_id;
  i = Math.max(0, rows.findIndex(row => row.normalized_id === nextId));
  const itemData = await api(`/api/item?id=${encodeURIComponent(rows[i].normalized_id)}`);
  item = itemData.item;
  boxes = structuredClone(item.boxes);
  rev = item.revision;
  sel = -1;
  undo = [];
  redo = [];
  $('note').value = item.note || '';
  $('pos').textContent = `${i + 1}/${rows.length}`;
  $('meta').innerHTML = [
    ['id', item.normalized_id],
    ['decision', item.decision || 'blank = implicit KEEP'],
    ['reasons', item.candidate_reasons],
    ['people', item.person_count],
    ['FN', item.false_negative_count],
  ].map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join('');
  im.onload = draw;
  im.src = `/api/${overlay ? 'overlay' : 'image'}?id=${encodeURIComponent(item.normalized_id)}`;
}

async function save() {
  if (!visible) throw new Error('BBox를 표시한 뒤 저장하세요.');
  const response = await fetch('/api/bbox', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({normalized_id: item.normalized_id, revision: rev, boxes}),
  });
  const data = await response.json();
  if (!data.ok) throw new Error(data.error);
  rev = data.value.revision;
  undo = [];
  redo = [];
  draw();
}

async function decision(value, moveNext = false) {
  const response = await fetch('/api/decision', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({normalized_id: item.normalized_id, decision: value, note: $('note').value}),
  });
  const data = await response.json();
  if (!data.ok) throw new Error(data.error);
  if (moveNext) i = Math.min(i + 1, rows.length - 1);
  await load();
}

$('canvas').onmousedown = event => {
  if (!visible) return;
  const point = canvasPoint(event);
  if (add) {
    drag = point;
    return;
  }
  const hits = boxes
    .map((box, index) => ({
      index,
      area: (box.x2 - box.x1) * (box.y2 - box.y1),
      hit: point.x >= box.x1 && point.x <= box.x2 && point.y >= box.y1 && point.y <= box.y2,
    }))
    .filter(candidate => candidate.hit)
    .sort((left, right) => left.area - right.area);
  sel = hits[0]?.index ?? -1;
  draw();
};

$('canvas').onmouseup = event => {
  if (!drag) return;
  const point = canvasPoint(event);
  const [x1, x2] = [drag.x, point.x].sort((a, b) => a - b);
  const [y1, y2] = [drag.y, point.y].sort((a, b) => a - b);
  drag = null;
  if (x2 - x1 >= 2 && y2 - y1 >= 2) {
    snapshot();
    boxes.push({class_id: 0, x1, y1, x2, y2});
    sel = boxes.length - 1;
  }
  add = false;
  draw();
};

document.addEventListener('keydown', event => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if (key === 'v') {
    visible = !visible;
    localStorage.bboxVisible = visible ? '1' : '0';
    draw();
    return;
  }
  if (!visible) return;
  if (key === 'a') {
    add = true;
    draw();
    return;
  }
  if (key === 'escape') {
    add = false;
    sel = -1;
    draw();
    return;
  }
  if (key === 's') {
    event.preventDefault();
    save().catch(error => alert(error.message));
    return;
  }
  if (key === 'w' && sel >= 0) {
    snapshot();
    boxes.splice(sel, 1);
    sel = -1;
    draw();
    return;
  }
  if (key === 'z' && event.ctrlKey && undo.length) {
    redo.push(JSON.stringify(boxes));
    boxes = JSON.parse(undo.pop());
    draw();
    return;
  }
  if ((key === 'y' || (key === 'z' && event.shiftKey)) && event.ctrlKey && redo.length) {
    undo.push(JSON.stringify(boxes));
    boxes = JSON.parse(redo.pop());
    draw();
    return;
  }
  if (!['arrowleft', 'arrowright', 'arrowup', 'arrowdown'].includes(key) || sel < 0) return;
  event.preventDefault();
  snapshot();
  const box = boxes[sel];
  const dx = key === 'arrowleft' ? -1 : key === 'arrowright' ? 1 : 0;
  const dy = key === 'arrowup' ? -1 : key === 'arrowdown' ? 1 : 0;
  if (event.ctrlKey) {
    if (dx < 0) box.x1 = Math.max(0, box.x1 - 1);
    if (dx > 0) box.x2 = Math.min(im.naturalWidth, box.x2 + 1);
    if (dy < 0) box.y1 = Math.max(0, box.y1 - 1);
    if (dy > 0) box.y2 = Math.min(im.naturalHeight, box.y2 + 1);
  } else {
    const width = box.x2 - box.x1;
    const height = box.y2 - box.y1;
    const x1 = Math.max(0, Math.min(im.naturalWidth - width, box.x1 + dx));
    const y1 = Math.max(0, Math.min(im.naturalHeight - height, box.y1 + dy));
    box.x1 = x1;
    box.y1 = y1;
    box.x2 = x1 + width;
    box.y2 = y1 + height;
  }
  draw();
});

$('add').onclick = () => { add = true; draw(); };
$('del').onclick = () => {
  if (sel >= 0) {
    snapshot();
    boxes.splice(sel, 1);
    sel = -1;
    draw();
  }
};
$('save').onclick = () => save().catch(error => alert(error.message));
$('reset').onclick = () => { boxes = structuredClone(item.boxes); sel = -1; undo = []; redo = []; draw(); };
$('vis').onclick = () => { visible = !visible; localStorage.bboxVisible = visible ? '1' : '0'; draw(); };
$('orig').onclick = () => { overlay = false; load().catch(error => alert(error.message)); };
$('over').onclick = () => { overlay = true; load().catch(error => alert(error.message)); };

load().catch(error => { $('state').textContent = error.message; });
