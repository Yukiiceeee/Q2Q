document.addEventListener('DOMContentLoaded', () => {
  AOS.init({
    duration: 600,
    easing: 'ease-out-cubic',
    once: true,
    offset: 80
  });

  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
});

function tabSwitch(group, idx) {
  document.querySelectorAll(`[data-tab-group="${group}"]`).forEach(el => {
    el.classList.add('hidden');
  });
  document.querySelectorAll(`[data-tab-btn="${group}"]`).forEach(btn => {
    btn.classList.remove('active');
  });
  const panel = document.querySelector(`[data-tab-group="${group}"][data-tab-idx="${idx}"]`);
  const btn = document.querySelector(`[data-tab-btn="${group}"][data-tab-idx="${idx}"]`);
  if (panel) panel.classList.remove('hidden');
  if (btn) btn.classList.add('active');
}

function openLightbox(src, title) {
  const overlay = document.getElementById('lightbox-overlay');
  const img = document.getElementById('lightbox-img');
  const caption = document.getElementById('lightbox-caption');
  img.src = src;
  caption.textContent = title;
  overlay.classList.remove('hidden');
  overlay.classList.add('flex');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  const overlay = document.getElementById('lightbox-overlay');
  overlay.classList.add('hidden');
  overlay.classList.remove('flex');
  document.body.style.overflow = '';
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeLightbox();
});
