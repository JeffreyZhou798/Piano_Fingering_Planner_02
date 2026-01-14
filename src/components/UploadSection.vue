<template>
  <section class="upload-section">
    <div class="upload-card">
      <div class="upload-icon">📄</div>
      <h2>Upload Your Piano Score</h2>
      <p class="description">
        Upload a MusicXML file (.mxl, .musicxml, or .xml) to generate intelligent fingering suggestions
      </p>
      
      <div 
        class="drop-zone"
        :class="{ 'drag-over': isDragging, 'processing': store.isProcessing }"
        @dragover.prevent="isDragging = true"
        @dragleave="isDragging = false"
        @drop.prevent="handleDrop"
        @click="triggerFileInput"
      >
        <input 
          ref="fileInput"
          type="file" 
          accept=".mxl,.musicxml,.xml"
          @change="handleFileSelect"
          hidden
        />
        
        <div v-if="store.isProcessing" class="processing-indicator">
          <div class="spinner"></div>
          <p>{{ store.processingStep || 'Processing...' }}</p>
        </div>
        
        <template v-else>
          <div class="drop-icon">📁</div>
          <p class="drop-text">
            <strong>Click to upload</strong> or drag and drop
          </p>
          <p class="file-types">MXL, MusicXML, XML files supported</p>
        </template>
      </div>
      
      <div class="features">
        <div class="feature">
          <span class="feature-icon">🎯</span>
          <span>Pattern Recognition</span>
        </div>
        <div class="feature">
          <span class="feature-icon">🧠</span>
          <span>AI-Powered Planning</span>
        </div>
        <div class="feature">
          <span class="feature-icon">✏️</span>
          <span>Manual Editing</span>
        </div>
        <div class="feature">
          <span class="feature-icon">💾</span>
          <span>Export Results</span>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useAppStore } from '../stores/appStore';

const store = useAppStore();
const fileInput = ref<HTMLInputElement | null>(null);
const isDragging = ref(false);

function triggerFileInput() {
  if (!store.isProcessing) {
    fileInput.value?.click();
  }
}

function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) {
    processFile(file);
  }
}

function handleDrop(event: DragEvent) {
  isDragging.value = false;
  const file = event.dataTransfer?.files[0];
  if (file) {
    processFile(file);
  }
}

async function processFile(file: File) {
  const validExtensions = ['.mxl', '.musicxml', '.xml'];
  const fileName = file.name.toLowerCase();
  
  if (!validExtensions.some(ext => fileName.endsWith(ext))) {
    store.addToast('error', 'Please upload a valid MusicXML file (.mxl, .musicxml, or .xml)');
    return;
  }
  
  await store.uploadFile(file);
}
</script>

<style scoped>
.upload-section {
  display: flex;
  justify-content: center;
  padding: 2rem 0;
}

.upload-card {
  background: var(--bg-secondary);
  border-radius: 16px;
  padding: 2.5rem;
  max-width: 600px;
  width: 100%;
  text-align: center;
  box-shadow: var(--shadow-lg);
}

.upload-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.upload-card h2 {
  font-size: 1.75rem;
  margin-bottom: 0.5rem;
  color: var(--text-primary);
}

.description {
  color: var(--text-secondary);
  margin-bottom: 2rem;
}

.drop-zone {
  border: 2px dashed var(--border-color);
  border-radius: 12px;
  padding: 3rem 2rem;
  cursor: pointer;
  transition: all 0.3s;
  background: var(--bg-primary);
}

.drop-zone:hover {
  border-color: var(--accent-color);
  background: var(--bg-hover);
}

.drop-zone.drag-over {
  border-color: var(--accent-color);
  background: var(--accent-light);
}

.drop-zone.processing {
  cursor: wait;
  opacity: 0.8;
}

.drop-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.drop-text {
  font-size: 1.1rem;
  color: var(--text-primary);
  margin-bottom: 0.5rem;
}

.file-types {
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.processing-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border-color);
  border-top-color: var(--accent-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.features {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-top: 2rem;
}

.feature {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  background: var(--bg-primary);
  border-radius: 8px;
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.feature-icon {
  font-size: 1.2rem;
}

@media (max-width: 500px) {
  .features {
    grid-template-columns: 1fr;
  }
}
</style>
