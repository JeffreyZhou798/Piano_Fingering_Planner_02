<template>
  <footer class="footer">
    <div class="footer-content">
      <p>Piano Fingering Planner - AI-powered fingering suggestions for piano scores</p>
      <p class="tech">Built with Vue 3 + TypeScript | Pattern Recognition + Dynamic Programming</p>
    </div>
  </footer>
</template>

<style scoped>
.footer {
  background: var(--bg-secondary);
  border-top: 1px solid var(--border-color);
  padding: 1.5rem 1rem;
  text-align: center;
}

.footer-content {
  max-width: 1200px;
  margin: 0 auto;
}

.footer p {
  margin: 0.25rem 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.footer .tech {
  font-size: 0.8rem;
  opacity: 0.7;
}
</style>
