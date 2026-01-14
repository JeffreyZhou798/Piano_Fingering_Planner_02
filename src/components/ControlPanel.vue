<template>
  <section class="control-panel">
    <div class="controls">
      <div class="control-group">
        <label>Difficulty Level</label>
        <div class="button-group">
          <button 
            v-for="level in levels" 
            :key="level.value"
            :class="{ active: store.difficultyLevel === level.value }"
            @click="store.setDifficultyLevel(level.value)"
          >
            {{ level.label }}
          </button>
        </div>
      </div>
      
      <div class="action-buttons">
        <button 
          class="btn btn-primary"
          :class="{ processing: store.isProcessing }"
          :disabled="store.isProcessing"
          @click="store.generateFingering()"
        >
          <span v-if="store.isProcessing" class="spinner-small"></span>
          <span v-else>🎯</span>
          {{ store.isProcessing ? store.processingStep : (store.hasFingering ? 'Regenerate' : 'Generate Fingering') }}
        </button>
        
        <button 
          class="btn btn-secondary"
          :disabled="!store.hasFingering"
          @click="store.downloadMusicXML()"
        >
          <span>💾</span>
          Download MusicXML
        </button>
      </div>
    </div>
    
    <div v-if="store.hasFingering" class="fingering-legend">
      <h4>Finger Legend</h4>
      <div class="legend-items">
        <div 
          v-for="finger in fingers" 
          :key="finger.num"
          class="legend-item"
          :style="{ color: store.colorMode ? finger.color : 'inherit' }"
        >
          <span class="finger-num" :style="{ backgroundColor: store.colorMode ? finger.color + '20' : 'var(--bg-primary)', borderColor: store.colorMode ? finger.color : 'var(--border-color)' }">
            {{ finger.num }}
          </span>
          <span class="finger-name">{{ finger.name }}</span>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { useAppStore } from '../stores/appStore';

const store = useAppStore();

const levels = [
  { value: 'beginner' as const, label: 'Beginner' },
  { value: 'intermediate' as const, label: 'Intermediate' },
  { value: 'advanced' as const, label: 'Advanced' }
];

const fingers = [
  { num: 1, name: 'Thumb', color: '#D32F2F' },
  { num: 2, name: 'Index', color: '#F57C00' },
  { num: 3, name: 'Middle', color: '#1976D2' },
  { num: 4, name: 'Ring', color: '#388E3C' },
  { num: 5, name: 'Pinky', color: '#7B1FA2' }
];
</script>

<style scoped>
.control-panel {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-md);
}

.controls {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  flex-wrap: wrap;
  gap: 1.5rem;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.control-group label {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--text-secondary);
}

.button-group {
  display: flex;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border-color);
}

.button-group button {
  padding: 0.5rem 1rem;
  border: none;
  background: var(--bg-primary);
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.9rem;
}

.button-group button:not(:last-child) {
  border-right: 1px solid var(--border-color);
}

.button-group button:hover {
  background: var(--bg-hover);
}

.button-group button.active {
  background: var(--accent-color);
  color: white;
}

.action-buttons {
  display: flex;
  gap: 1rem;
}

.btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
}

.btn-primary {
  background: var(--accent-color);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: var(--accent-dark);
}

.btn-primary.processing {
  background: var(--accent-light);
}

.btn-secondary {
  background: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--bg-hover);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.spinner-small {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.fingering-legend {
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border-color);
}

.fingering-legend h4 {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--text-secondary);
  margin: 0 0 0.75rem 0;
}

.legend-items {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.finger-num {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  font-size: 0.85rem;
  font-weight: 600;
  border: 1px solid;
}

.finger-name {
  font-size: 0.85rem;
  color: var(--text-secondary);
}

@media (max-width: 700px) {
  .controls {
    flex-direction: column;
    align-items: stretch;
  }
  
  .action-buttons {
    flex-direction: column;
  }
  
  .btn {
    justify-content: center;
  }
}
</style>
