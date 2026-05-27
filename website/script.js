(function () {
  var year = document.getElementById('year');
  if (year) year.textContent = new Date().getFullYear();

  var path = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.tab').forEach(function (tab) {
    var href = tab.getAttribute('href');
    tab.classList.toggle('active', href === path);
  });
})();
