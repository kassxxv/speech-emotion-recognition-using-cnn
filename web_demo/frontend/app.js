'use strict';

class WaveformDrawer {
  constructor(canvasId, analyser) {
    this.canvas = document.getElementById(canvasId);
    // Sync canvas pixel dimensions to its CSS display size for sharp rendering
    this.canvas.width  = this.canvas.offsetWidth  || 780;
    this.canvas.height = this.canvas.offsetHeight || 100;
    this.ctx2d  = this.canvas.getContext('2d');
    this.analyser = analyser;
    this.buffer = new Float32Array(analyser.fftSize);
    this._animId = null;
  }

  start() {
    const draw = () => {
      this._animId = requestAnimationFrame(draw);
      this.analyser.getFloatTimeDomainData(this.buffer);

      const c = this.ctx2d;
      const W = this.canvas.width;
      const H = this.canvas.height;

      c.clearRect(0, 0, W, H);
      c.beginPath();
      c.strokeStyle = '#f472b6';
      c.lineWidth = 1.5;

      for (let i = 0; i < this.buffer.length; i++) {
        const x = (i / (this.buffer.length - 1)) * W;
        const y = ((this.buffer[i] + 1) / 2) * H;
        i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
      }
      c.stroke();
    };
    draw();
  }

  stop() {
    if (this._animId) { cancelAnimationFrame(this._animId); this._animId = null; }
    const c = this.ctx2d;
    c.clearRect(0, 0, this.canvas.width, this.canvas.height);
    c.beginPath();
    c.strokeStyle = '#222230';
    c.lineWidth = 1;
    c.moveTo(0, this.canvas.height / 2);
    c.lineTo(this.canvas.width, this.canvas.height / 2);
    c.stroke();
  }
}


class EmotionBars {
  constructor(containerId, smoothing = 2) {
    this.container = document.getElementById(containerId);
    this._rows   = {};
    this._window = [];       // rolling buffer of prob arrays
    this._smooth = smoothing; // number of predictions to average
    this._build();
  }

  _build() {
    const emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad'];
    emotions.forEach(name => {
      const row   = document.createElement('div');
      row.className = 'bar-row';
      row.dataset.emotion = name;

      const label = document.createElement('span');
      label.className = 'bar-label';
      label.textContent = name;

      const track = document.createElement('div');
      track.className = 'bar-track';
      const fill = document.createElement('div');
      fill.className = 'bar-fill';
      track.appendChild(fill);

      const pct = document.createElement('span');
      pct.className = 'bar-pct';
      pct.textContent = '0%';

      row.append(label, track, pct);
      this.container.appendChild(row);
      this._rows[name] = { row, fill, pct };
    });
  }

  update({ emotion, probs, emotions }) {
    // Rolling average over last N predictions
    this._window.push(probs);
    if (this._window.length > this._smooth) this._window.shift();

    const avg = probs.map((_, i) =>
      this._window.reduce((s, p) => s + p[i], 0) / this._window.length
    );
    const dominantIdx = avg.indexOf(Math.max(...avg));
    const dominantName = emotions[dominantIdx];

    emotions.forEach((name, i) => {
      const entry = this._rows[name];
      if (!entry) return;
      const p = (avg[i] * 100).toFixed(1);
      entry.fill.style.width = p + '%';
      entry.pct.textContent  = p + '%';
      entry.row.classList.toggle('dominant', name === dominantName);
    });
    document.getElementById('current-emotion').textContent = dominantName;
  }
}


class AudioRecorder {
  constructor(onChunk) {
    this.onChunk   = onChunk;
    this.analyser  = null;
    this._ctx      = null;
    this._stream   = null;
    this._processor = null;
    this._acc      = new Float32Array(0);
    this._chunkLen = 0;
  }

  get sampleRate() { return this._ctx ? this._ctx.sampleRate : 16000; }

  async start() {
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    this._ctx    = new AudioContext({ sampleRate: 16000 });
    this._chunkLen = this._ctx.sampleRate * 2;  // 2-second chunks

    const source = this._ctx.createMediaStreamSource(this._stream);

    this.analyser = this._ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    source.connect(this.analyser);

    this._processor = this._ctx.createScriptProcessor(4096, 1, 1);
    source.connect(this._processor);

    const silence = this._ctx.createGain();
    silence.gain.value = 0;
    this._processor.connect(silence);
    silence.connect(this._ctx.destination);

    this._acc = new Float32Array(0);
    this._processor.onaudioprocess = (e) => {
      const samples = e.inputBuffer.getChannelData(0);
      const merged  = new Float32Array(this._acc.length + samples.length);
      merged.set(this._acc);
      merged.set(samples, this._acc.length);
      this._acc = merged;

      if (this._acc.length >= this._chunkLen) {
        const chunk = this._acc.slice(0, this._chunkLen);
        this._acc   = this._acc.slice(this._chunkLen);
        this.onChunk(chunk);
      }
    };
  }

  stop() {
    if (this._processor) { this._processor.disconnect(); this._processor = null; }
    if (this._stream)    { this._stream.getTracks().forEach(t => t.stop()); this._stream = null; }
    if (this._ctx)       { this._ctx.close(); this._ctx = null; }
    this._acc = new Float32Array(0);
    this.analyser = null;
  }
}


// ── Main ──────────────────────────────────────────────────────────────────────

const emotionBars = new EmotionBars('emotion-bars');
const toggleBtn   = document.getElementById('toggle');
const statusEl    = document.getElementById('status');

let recorder = null;
let drawer   = null;
let ws       = null;
let active   = false;

toggleBtn.addEventListener('click', () => {
  active ? stop() : start().catch(err => {
    statusEl.textContent = 'Error: ' + err.message;
    stop();
  });
});

async function start() {
  statusEl.textContent = 'Connecting…';

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error('WebSocket failed')); });

  recorder = new AudioRecorder((chunk) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(chunk.buffer);
  });

  await recorder.start();
  ws.send(JSON.stringify({ type: 'config', sample_rate: recorder.sampleRate }));

  drawer = new WaveformDrawer('waveform', recorder.analyser);
  drawer.start();

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.silent) {
      statusEl.textContent = 'Listening… (speak louder or closer)';
      return;
    }
    statusEl.textContent = 'Recording — speak now';
    emotionBars.update(data);
  };
  ws.onclose   = () => { if (active) stop(); };

  active = true;
  toggleBtn.textContent = 'Stop Recording';
  toggleBtn.classList.add('recording');
  statusEl.textContent = 'Recording — speak now';
}

function stop() {
  if (recorder) { recorder.stop(); recorder = null; }
  if (drawer)   { drawer.stop();   drawer   = null; }
  if (ws)       { ws.close();      ws       = null; }
  emotionBars._window = [];  // clear smoothing history
  active = false;
  toggleBtn.textContent = 'Start Recording';
  toggleBtn.classList.remove('recording');
  statusEl.textContent = 'Stopped';
}
