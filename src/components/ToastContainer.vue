<template>
  <div class="toast-container">
    <TransitionGroup name="toast">
      <div 
        v-for="toast in store.toasts" 
        :key="toast.id"
        :class="['toast', `toast-${toast.type}`]"
      >
        <span class="toast-icon">{{ getIcon(toast.type) }}</span>
        <span class="toast-message">{{ toast.message }}</span>
        <button class="toast-close" @click="store.removeToast(toast.id)">✕</button>
      </div>
    </TransitionGroup>
  </div>
</template>

<script setup lang="ts">
import { useAppStore } from '../stores/appStore';
import type { Toast } from '../types';

const store = useAppStore();

function getIcon(type: Toast['type']): string {
  const icons: Record<Toast['type'], string> = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ'
  };
  return icons[type];
}
</script>

<style scoped>
.toast-container {
  position: fixed;
  bottom: 1.5rem;
  right: 1.5rem;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  max-width: 400px;
}

.toast {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem 1.25rem;
  border-radius: 8px;
  background: var(--bg-secondary);
  box-shadow: var(--shadow-lg);
  border-left: 4px solid;
}

.toast-success {
  border-color: #4CAF50;
}

.toast-success .toast-icon {
  color: #4CAF50;
}

.toast-error {
  border-color: #F44336;
}

.toast-error .toast-icon {
  color: #F44336;
}

.toast-warning {
  border-color: #FF9800;
}

.toast-warning .toast-icon {
  color: #FF9800;
}

.toast-info {
  border-color: #2196F3;
}

.toast-info .toast-icon {
  color: #2196F3;
}

.toast-icon {
  font-size: 1.25rem;
  font-weight: bold;
}

.toast-message {
  flex: 1;
  font-size: 0.95rem;
  color: var(--text-primary);
}

.toast-close {
  background: none;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  padding: 0.25rem;
  font-size: 0.9rem;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.toast-close:hover {
  opacity: 1;
}

/* Transitions */
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(100%);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100%);
}

@media (max-width: 500px) {
  .toast-container {
    left: 1rem;
    right: 1rem;
    max-width: none;
  }
}
</style>
