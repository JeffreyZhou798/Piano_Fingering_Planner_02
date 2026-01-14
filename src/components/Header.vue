<template>
  <header class="header">
    <div class="header-content">
      <div class="logo">
        <span class="logo-icon">🎹</span>
        <h1>Piano Fingering Planner</h1>
      </div>
      <nav class="nav">
        <button 
          class="nav-btn"
          :class="{ active: store.colorMode }"
          @click="store.toggleColorMode()"
          title="Toggle color mode"
        >
          <span class="icon">🎨</span>
          <span class="label">Colors</span>
        </button>
        <button 
          class="nav-btn"
          :class="{ active: store.darkMode }"
          @click="store.toggleDarkMode()"
          title="Toggle dark mode"
        >
          <span class="icon">{{ store.darkMode ? '☀️' : '🌙' }}</span>
          <span class="label">{{ store.darkMode ? 'Light' : 'Dark' }}</span>
        </button>
      </nav>
    </div>
  </header>
</template>

<script setup lang="ts">
import { useAppStore } from '../stores/appStore';

const store = useAppStore();
</script>

<style scoped>
.header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
  padding: 1rem;
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.logo-icon {
  font-size: 2rem;
}

.logo h1 {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.nav {
  display: flex;
  gap: 0.5rem;
}

.nav-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--bg-primary);
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.2s;
}

.nav-btn:hover {
  background: var(--bg-hover);
}

.nav-btn.active {
  background: var(--accent-color);
  color: white;
  border-color: var(--accent-color);
}

.nav-btn .icon {
  font-size: 1.1rem;
}

.nav-btn .label {
  font-size: 0.9rem;
}

@media (max-width: 600px) {
  .logo h1 {
    font-size: 1.1rem;
  }
  
  .nav-btn .label {
    display: none;
  }
}
</style>
