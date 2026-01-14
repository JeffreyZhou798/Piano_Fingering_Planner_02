<template>
  <section class="score-viewer">
    <div class="viewer-header">
      <h3>Fingering Results</h3>
      <div class="view-controls">
        <button 
          :class="{ active: viewMode === 'list' }"
          @click="viewMode = 'list'"
        >
          📋 List View
        </button>
        <button 
          :class="{ active: viewMode === 'measure' }"
          @click="viewMode = 'measure'"
        >
          📊 By Measure
        </button>
      </div>
    </div>
    
    <div class="viewer-content">
      <!-- List View -->
      <div v-if="viewMode === 'list'" class="list-view">
        <div class="hand-section">
          <h4>Right Hand (RH)</h4>
          <div class="notes-grid">
            <NoteCard 
              v-for="(note, idx) in store.rhNotes" 
              :key="note.id"
              :note="note"
              :index="getGlobalIndex(note)"
              @update="handleFingeringUpdate"
            />
          </div>
        </div>
        
        <div class="hand-section">
          <h4>Left Hand (LH)</h4>
          <div class="notes-grid">
            <NoteCard 
              v-for="(note, idx) in store.lhNotes" 
              :key="note.id"
              :note="note"
              :index="getGlobalIndex(note)"
              @update="handleFingeringUpdate"
            />
          </div>
        </div>
      </div>
      
      <!-- Measure View -->
      <div v-else class="measure-view">
        <div 
          v-for="measure in groupedByMeasure" 
          :key="measure.number"
          class="measure-card"
        >
          <div class="measure-header">
            <span class="measure-num">M. {{ measure.number }}</span>
          </div>
          <div class="measure-content">
            <div v-if="measure.rhNotes.length > 0" class="measure-hand">
              <span class="hand-label">RH:</span>
              <div class="measure-notes">
                <span 
                  v-for="note in measure.rhNotes" 
                  :key="note.id"
                  class="note-fingering"
                  :style="getFingeringStyle(note.fingering)"
                  @click="openEditor(note)"
                >
                  {{ note.fingering || '-' }}
                </span>
              </div>
            </div>
            <div v-if="measure.lhNotes.length > 0" class="measure-hand">
              <span class="hand-label">LH:</span>
              <div class="measure-notes">
                <span 
                  v-for="note in measure.lhNotes" 
                  :key="note.id"
                  class="note-fingering"
                  :style="getFingeringStyle(note.fingering)"
                  @click="openEditor(note)"
                >
                  {{ note.fingering || '-' }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Edit Modal -->
    <div v-if="editingNote" class="modal-overlay" @click.self="editingNote = null">
      <div class="modal">
        <h4>Edit Fingering</h4>
        <p class="note-info">
          {{ getNoteName(editingNote) }} ({{ editingNote.hand }}) - Measure {{ editingNote.measureNumber }}
        </p>
        <div class="finger-buttons">
          <button 
            v-for="f in 5" 
            :key="f"
            :class="{ active: editingNote.fingering === f }"
            :style="getFingerButtonStyle(f as 1|2|3|4|5)"
            @click="updateFingering(f as 1|2|3|4|5)"
          >
            {{ f }}
          </button>
        </div>
        <button class="close-btn" @click="editingNote = null">Close</button>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useAppStore } from '../stores/appStore';
import type { Note, Finger } from '../types';
import NoteCard from './NoteCard.vue';

const store = useAppStore();
const viewMode = ref<'list' | 'measure'>('measure');
const editingNote = ref<Note | null>(null);

const fingerColors: Record<Finger, string> = {
  1: '#D32F2F',
  2: '#F57C00',
  3: '#1976D2',
  4: '#388E3C',
  5: '#7B1FA2'
};

const groupedByMeasure = computed(() => {
  if (!store.parsedScore) return [];
  
  const measures: { number: number; rhNotes: Note[]; lhNotes: Note[] }[] = [];
  const measureMap = new Map<number, { rhNotes: Note[]; lhNotes: Note[] }>();
  
  store.parsedScore.notes.forEach(note => {
    if (!measureMap.has(note.measureNumber)) {
      measureMap.set(note.measureNumber, { rhNotes: [], lhNotes: [] });
    }
    const m = measureMap.get(note.measureNumber)!;
    if (note.hand === 'RH') {
      m.rhNotes.push(note);
    } else {
      m.lhNotes.push(note);
    }
  });
  
  measureMap.forEach((value, key) => {
    measures.push({ number: key, ...value });
  });
  
  return measures.sort((a, b) => a.number - b.number);
});

function getGlobalIndex(note: Note): number {
  return store.parsedScore?.notes.indexOf(note) ?? -1;
}

function getNoteName(note: Note): string {
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const pitchClass = note.pitch % 12;
  const octave = Math.floor(note.pitch / 12) - 1;
  return `${noteNames[pitchClass]}${octave}`;
}

function getFingeringStyle(finger?: Finger) {
  if (!finger || !store.colorMode) return {};
  return {
    color: fingerColors[finger],
    borderColor: fingerColors[finger],
    backgroundColor: fingerColors[finger] + '15'
  };
}

function getFingerButtonStyle(finger: Finger) {
  if (!store.colorMode) return {};
  return {
    '--finger-color': fingerColors[finger]
  };
}

function openEditor(note: Note) {
  editingNote.value = note;
}

function updateFingering(finger: Finger) {
  if (editingNote.value) {
    const index = getGlobalIndex(editingNote.value);
    if (index >= 0) {
      store.updateNoteFingering(index, finger);
      editingNote.value.fingering = finger;
    }
  }
}

function handleFingeringUpdate(index: number, finger: Finger) {
  store.updateNoteFingering(index, finger);
}
</script>

<style scoped>
.score-viewer {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: var(--shadow-md);
}

.viewer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.viewer-header h3 {
  margin: 0;
  font-size: 1.25rem;
  color: var(--text-primary);
}

.view-controls {
  display: flex;
  gap: 0.5rem;
}

.view-controls button {
  padding: 0.5rem 1rem;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: var(--bg-primary);
  color: var(--text-primary);
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}

.view-controls button:hover {
  background: var(--bg-hover);
}

.view-controls button.active {
  background: var(--accent-color);
  color: white;
  border-color: var(--accent-color);
}

.hand-section {
  margin-bottom: 2rem;
}

.hand-section h4 {
  font-size: 1rem;
  color: var(--text-secondary);
  margin: 0 0 1rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-color);
}

.notes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
  gap: 0.75rem;
}

.measure-view {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.measure-card {
  background: var(--bg-primary);
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border-color);
}

.measure-header {
  background: var(--bg-hover);
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid var(--border-color);
}

.measure-num {
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.measure-content {
  padding: 0.75rem;
}

.measure-hand {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.measure-hand:last-child {
  margin-bottom: 0;
}

.hand-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-secondary);
  min-width: 28px;
}

.measure-notes {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
}

.note-fingering {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.note-fingering:hover {
  transform: scale(1.1);
  box-shadow: var(--shadow-sm);
}

/* Modal */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal {
  background: var(--bg-secondary);
  border-radius: 12px;
  padding: 1.5rem;
  min-width: 300px;
  max-width: 90vw;
}

.modal h4 {
  margin: 0 0 0.5rem 0;
  font-size: 1.25rem;
  color: var(--text-primary);
}

.note-info {
  color: var(--text-secondary);
  margin-bottom: 1rem;
}

.finger-buttons {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.finger-buttons button {
  flex: 1;
  padding: 1rem;
  font-size: 1.25rem;
  font-weight: 600;
  border: 2px solid var(--border-color);
  border-radius: 8px;
  background: var(--bg-primary);
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.2s;
}

.finger-buttons button:hover {
  background: var(--bg-hover);
}

.finger-buttons button.active {
  background: var(--accent-color);
  color: white;
  border-color: var(--accent-color);
}

.close-btn {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--bg-primary);
  color: var(--text-primary);
  cursor: pointer;
  font-size: 1rem;
}

.close-btn:hover {
  background: var(--bg-hover);
}
</style>
