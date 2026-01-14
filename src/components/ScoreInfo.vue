<template>
  <section class="score-info" v-if="store.parsedScore">
    <div class="info-card">
      <div class="info-header">
        <div class="title-section">
          <h2>{{ store.parsedScore.title }}</h2>
          <p class="composer">{{ store.parsedScore.composer }}</p>
        </div>
        <button class="reset-btn" @click="store.reset()" title="Upload new file">
          <span>✕</span>
        </button>
      </div>
      
      <div class="stats">
        <div class="stat">
          <span class="stat-value">{{ store.noteCount }}</span>
          <span class="stat-label">Notes</span>
        </div>
        <div class="stat">
          <span class="stat-value">{{ store.measureCount }}</span>
          <span class="stat-label">Measures</span>
        </div>
        <div class="stat">
          <span class="stat-value">{{ store.rhNotes.length }}</span>
          <span class="stat-label">RH Notes</span>
        </div>
        <div class="stat">
          <span class="stat-value">{{ store.lhNotes.length }}</span>
          <span class="stat-label">LH Notes</span>
        </div>
      </div>
      
      <div v-if="Object.keys(store.patternSummary).length > 0" class="patterns">
        <h3>Detected Patterns</h3>
        <div class="pattern-tags">
          <span 
            v-for="(count, type) in store.patternSummary" 
            :key="type"
            class="pattern-tag"
            :class="getPatternClass(type as string)"
          >
            {{ type }}: {{ count }}
          </span>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { useAppStore } from '../stores/appStore';

const store = useAppStore();

function getPatternClass(type: string): string {
  const classes: Record<string, string> = {
    'SCALE': 'pattern-scale',
    'ARPEGGIO': 'pattern-arpeggio',
    'CHORDAL': 'pattern-chordal',
    'REPEATED': 'pattern-repeated',
    'LEAP': 'pattern-leap',
    'MELODIC': 'pattern-melodic',
    'ALBERTI': 'pattern-alberti',
    'ORNAMENTED': 'pattern-ornamented',
    'OSTINATO': 'pattern-ostinato',
    'POLYPHONIC': 'pattern-polyphonic',
    'UNKNOWN': 'pattern-unknown'
  };
  return classes[type] || 'pattern-unknown';
}
</script>

<style scoped>
.score-info {
  margin-bottom: 1.5rem;
}

.info-card {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: var(--shadow-md);
}

.info-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1.5rem;
}

.title-section h2 {
  font-size: 1.5rem;
  margin: 0 0 0.25rem 0;
  color: var(--text-primary);
}

.composer {
  color: var(--text-secondary);
  font-size: 1rem;
  margin: 0;
}

.reset-btn {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: 1px solid var(--border-color);
  background: var(--bg-primary);
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.reset-btn:hover {
  background: var(--danger-color);
  color: white;
  border-color: var(--danger-color);
}

.stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.stat {
  text-align: center;
  padding: 1rem;
  background: var(--bg-primary);
  border-radius: 8px;
}

.stat-value {
  display: block;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--accent-color);
}

.stat-label {
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.patterns h3 {
  font-size: 1rem;
  margin: 0 0 0.75rem 0;
  color: var(--text-primary);
}

.pattern-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.pattern-tag {
  padding: 0.35rem 0.75rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

.pattern-scale { background: #E3F2FD; color: #1565C0; }
.pattern-arpeggio { background: #F3E5F5; color: #7B1FA2; }
.pattern-chordal { background: #E8F5E9; color: #2E7D32; }
.pattern-repeated { background: #FFF3E0; color: #E65100; }
.pattern-leap { background: #FFEBEE; color: #C62828; }
.pattern-melodic { background: #E0F7FA; color: #00838F; }
.pattern-alberti { background: #FBE9E7; color: #BF360C; }
.pattern-ornamented { background: #F9FBE7; color: #827717; }
.pattern-ostinato { background: #ECEFF1; color: #37474F; }
.pattern-polyphonic { background: #EDE7F6; color: #4527A0; }
.pattern-unknown { background: #FAFAFA; color: #616161; }

.dark .pattern-tag {
  filter: brightness(0.8);
}

@media (max-width: 600px) {
  .stats {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
