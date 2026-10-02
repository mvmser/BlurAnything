// Loaded synchronously from <head>: picks the theme before first paint so the
// page never flashes the wrong one (saved choice, else the system's).
(function () {
  var theme = null;
  try {
    theme = localStorage.getItem('ba-theme');
  } catch (e) {
    /* storage blocked */
  }
  if (!theme) theme = window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.dataset.theme = theme;
})();
