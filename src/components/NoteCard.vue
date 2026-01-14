<template>
  <div class="note-card" @click="showEditor = true">
    <div class="note-pitch">{{ noteName }}</div>
    <div 
      class="note-fingering"
      :style="fingeringStyle"
    >
      {{ note.fingering || '?' }}
    </div>
    <div class="note-meta">
      M.{{ note.measureNumber }}
    </div>
    
    <!-- Inline Editor -->
    <div v-if="showEditor" class="editor-popup" @click.stop>
      <div class="editor-buttons">
        <button 
          v-for="f in 5" 
          :key="f"
          :class="{ active: note.fingering === f }"
          @click="selectFinger(f as 1|2|3|4|5)"
        >
          {{ f }}
        </button>
      </div>
      <button class="close-editor" @click.stop="showEditor = false">✕</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useAppStore } from '../stores/appStore';
import type { Note, Finger } from '../types';

const props = defineProps<{
  note: Note;
  index: number;
}>();

const emit = defineEmits<{
  update: [index: number, finger: Finger];
}>();

const store = useAppStore();
const showEditor = ref(false);

const fingerColors: Record<Finger, string> = {
  1: '#D32F2F',
  2: '#F57C00',
  3: '#1976D2',
  4: '#388E3C',
  5: '#7B1FA2'
};

const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const noteName = computed(() => {
  const pitchClass = props.note.pitch % 12;
  const octave = Math.floor(props.note.pitch / 12) - 1;
  return `${noteNames[pitchClass]}${octave}`;
});

const fingeringStyle = computed(() => {
  const finger = props.note.fingering;
  if (!finger || !store.colorMode) return {};
  return {
    color: fingerColors[finger],
    borderColor: fingerColors[finger],
    backgroundColor: fingerColors[finger] + '20'
  };
});

function selectFinger(finger: Finger) {
  emit('update', props.index, finger);
  showEditor.value = false;
}
</script>

<style scoped>
.note-card {
  position: relative;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 0.5rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}

.note-card:hover {
  border-color: var(--accent-color);
  box-shadow: var(--shadow-sm);
}

.note-pitch {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
}

.note-fingering {
  font-size: 1.25rem;
  font-weight: 700;
  padding: 0.25rem;
  border: 2px solid var(--border-color);
  border-radius: 6px;
  margin-bottom: 0.25rem;
}

.note-meta {
  font-size: 0.7rem;
  color: var(--text-secondary);
}

.editor-popup {
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 0.5rem;
  box-shadow: var(--shadow-lg);
  z-index: 100;
  margin-top: 0.25rem;
}

.editor-buttons {
  display: flex;
  gap: 0.25rem;
}

.editor-buttons button {
  width: 32px;
  height: 32px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.editor-buttons button:hover {
  background: var(--bg-hover);
}

.editor-buttons button.active {
  background: var(--accent-color);
  color: white;
  border-color: var(--accent-color);
}

.close-editor {
  position: absolute;
  top: -8px;
  right: -8px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  border: 1px solid var(--border-color);
  background: var(--bg-secondary);
  color: var(--text-secondary);
  font-size: 0.7rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-editor:hover {
  background: var(--danger-color);
  color: white;
  border-color: var(--danger-color);
}
</style>
