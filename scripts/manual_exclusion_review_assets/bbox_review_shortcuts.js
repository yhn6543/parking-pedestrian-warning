function isEditingText(event) {
  return ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName);
}

async function saveBeforeLeaving() {
  if (!isDirty()) return true;
  if (!confirm('Unsaved BBox edits exist. Save before moving?')) return false;
  await save();
  return true;
}

async function guardedMove(delta) {
  if (!await saveBeforeLeaving()) return;
  i = Math.max(0, Math.min(rows.length - 1, i + delta));
  await load();
}

async function guardedDecision(value, moveNext = false) {
  if (!await saveBeforeLeaving()) return;
  await decision(value, moveNext);
}

function runShortcut(action) {
  action().catch(error => alert(error.message));
}

document.addEventListener('keydown', event => {
  if (isEditingText(event)) return;
  const key = event.key.toLowerCase();
  if (key === 'k') {
    event.preventDefault();
    runShortcut(() => guardedDecision('KEEP'));
  } else if (key === 'f') {
    event.preventDefault();
    runShortcut(() => guardedDecision('DROP'));
  } else if (key === 'c') {
    event.preventDefault();
    runShortcut(() => guardedDecision('HOLD'));
  } else if (key === 'q') {
    event.preventDefault();
    runShortcut(() => guardedMove(-1));
  } else if (key === 'e') {
    event.preventDefault();
    runShortcut(() => guardedMove(1));
  } else if (event.code === 'Space') {
    event.preventDefault();
    runShortcut(() => guardedDecision('DROP', true));
  }
});

window.addEventListener('beforeunload', event => {
  if (!isDirty()) return;
  event.preventDefault();
  event.returnValue = '';
});

$('prev').onclick = () => runShortcut(() => guardedMove(-1));
$('next').onclick = () => runShortcut(() => guardedMove(1));
$('keep').onclick = () => runShortcut(() => guardedDecision('KEEP'));
$('drop').onclick = () => runShortcut(() => guardedDecision('DROP'));
$('hold').onclick = () => runShortcut(() => guardedDecision('HOLD'));
