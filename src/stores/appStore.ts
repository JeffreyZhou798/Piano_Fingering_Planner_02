import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import type { 
  ParsedScore, PatternSegment, FingeringSolution, 
  FingeringPlacement, Toast, Finger, Note 
} from '@/types';
import { 
  musicXMLParser, 
  patternRecognizer, 
  fingeringPlanner, 
  fingeringRenderer 
} from '@/core';

export const useAppStore = defineStore('app', () => {
  // State
  const originalXml = ref<string | null>(null);
  const fileName = ref<string>('');
  const parsedScore = ref<ParsedScore | null>(null);
  const patterns = ref<PatternSegment[]>([]);
  const fingeringSolution = ref<FingeringSolution | null>(null);
  const placements = ref<FingeringPlacement[]>([]);
  const isProcessing = ref(false);
  const processingStep = ref('');
  const error = ref<string | null>(null);
  const darkMode = ref(false);
  const colorMode = ref(false);
  const difficultyLevel = ref<'beginner' | 'intermediate' | 'advanced'>('intermediate');
  const toasts = ref<Toast[]>([]);
  
  // Computed
  const hasScore = computed(() => parsedScore.value !== null);
  const hasFingering = computed(() => fingeringSolution.value !== null);
  const noteCount = computed(() => parsedScore.value?.notes.length || 0);
  const measureCount = computed(() => parsedScore.value?.measures.length || 0);
  
  const rhNotes = computed(() => 
    parsedScore.value?.notes.filter(n => n.hand === 'RH') || []
  );
  
  const lhNotes = computed(() => 
    parsedScore.value?.notes.filter(n => n.hand === 'LH') || []
  );
  
  const patternSummary = computed(() => {
    const summary: Record<string, number> = {};
    patterns.value.forEach(p => {
      summary[p.patternType] = (summary[p.patternType] || 0) + 1;
    });
    return summary;
  });

  // Actions
  async function uploadFile(file: File) {
    isProcessing.value = true;
    processingStep.value = 'Parsing MusicXML...';
    error.value = null;
    
    try {
      // Read and parse file
      const xmlContent = await readFileContent(file);
      originalXml.value = xmlContent;
      fileName.value = file.name;
      
      // Parse MusicXML
      const parsed = musicXMLParser.parseXML(xmlContent);
      parsedScore.value = parsed;
      
      addToast('success', `Successfully loaded "${parsed.title}" with ${parsed.notes.length} notes`);
      
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to parse file';
      error.value = message;
      addToast('error', message);
    } finally {
      isProcessing.value = false;
      processingStep.value = '';
    }
  }

  async function readFileContent(file: File): Promise<string> {
    const fileName = file.name.toLowerCase();
    
    if (fileName.endsWith('.mxl')) {
      // Handle compressed MXL
      const JSZip = (await import('jszip')).default;
      const zip = new JSZip();
      const contents = await zip.loadAsync(file);
      
      // Find container.xml
      const containerFile = contents.file('META-INF/container.xml');
      if (containerFile) {
        const containerXml = await containerFile.async('string');
        const parser = new DOMParser();
        const containerDoc = parser.parseFromString(containerXml, 'text/xml');
        const rootFile = containerDoc.querySelector('rootfile');
        if (rootFile) {
          const fullPath = rootFile.getAttribute('full-path');
          if (fullPath) {
            const musicXmlFile = contents.file(fullPath);
            if (musicXmlFile) {
              return musicXmlFile.async('string');
            }
          }
        }
      }
      
      // Fallback: find any XML file
      for (const [path, zipEntry] of Object.entries(contents.files)) {
        if (path.endsWith('.xml') && !path.startsWith('META-INF') && !zipEntry.dir) {
          return (zipEntry as any).async('string');
        }
      }
      
      throw new Error('Could not find MusicXML content in MXL file');
    }
    
    return file.text();
  }

  async function generateFingering() {
    if (!parsedScore.value) {
      addToast('error', 'Please upload a score first');
      return;
    }
    
    isProcessing.value = true;
    error.value = null;
    
    try {
      // Step 1: Pattern Recognition
      processingStep.value = 'Analyzing musical patterns...';
      await delay(100); // Allow UI update
      
      const recognizedPatterns = patternRecognizer.recognizePatterns(parsedScore.value.notes);
      patterns.value = recognizedPatterns;
      
      // Step 2: Fingering Planning
      processingStep.value = 'Planning optimal fingering...';
      await delay(100);
      
      fingeringPlanner.setDifficultyLevel(difficultyLevel.value);
      const solution = fingeringPlanner.planFingering(parsedScore.value.notes, recognizedPatterns);
      fingeringSolution.value = solution;
      
      // Apply fingering to notes
      parsedScore.value.notes.forEach((note, i) => {
        note.fingering = solution.fingering[i];
      });
      
      // Step 3: Calculate placements
      processingStep.value = 'Calculating display positions...';
      await delay(100);
      
      fingeringRenderer.setColorMode(colorMode.value);
      placements.value = fingeringRenderer.calculatePlacements(parsedScore.value.notes);
      
      addToast('success', `Generated fingering for ${parsedScore.value.notes.length} notes`);
      
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to generate fingering';
      error.value = message;
      addToast('error', message);
    } finally {
      isProcessing.value = false;
      processingStep.value = '';
    }
  }

  function updateNoteFingering(noteIndex: number, finger: Finger) {
    if (!parsedScore.value || !fingeringSolution.value) return;
    
    const note = parsedScore.value.notes[noteIndex];
    if (note) {
      note.fingering = finger;
      fingeringSolution.value.fingering[noteIndex] = finger;
      
      // Recalculate placements
      fingeringRenderer.setColorMode(colorMode.value);
      placements.value = fingeringRenderer.calculatePlacements(parsedScore.value.notes);
      
      addToast('info', `Updated note ${noteIndex + 1} to finger ${finger}`);
    }
  }

  function downloadMusicXML() {
    if (!originalXml.value || !parsedScore.value) {
      addToast('error', 'No score to download');
      return;
    }
    
    try {
      const annotatedXml = musicXMLParser.writeFingeringToXML(
        originalXml.value, 
        parsedScore.value.notes
      );
      
      const blob = new Blob([annotatedXml], { type: 'application/xml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName.value.replace(/\.(mxl|musicxml|xml)$/i, '_fingered.musicxml');
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      addToast('success', 'Downloaded annotated MusicXML file');
    } catch (err) {
      addToast('error', 'Failed to download file');
    }
  }

  function toggleDarkMode() {
    darkMode.value = !darkMode.value;
    document.documentElement.classList.toggle('dark', darkMode.value);
  }

  function toggleColorMode() {
    colorMode.value = !colorMode.value;
    fingeringRenderer.setColorMode(colorMode.value);
    
    if (parsedScore.value) {
      placements.value = fingeringRenderer.calculatePlacements(parsedScore.value.notes);
    }
  }

  function setDifficultyLevel(level: 'beginner' | 'intermediate' | 'advanced') {
    difficultyLevel.value = level;
  }

  function addToast(type: Toast['type'], message: string, duration = 4000) {
    const id = Date.now().toString();
    toasts.value.push({ id, type, message, duration });
    
    setTimeout(() => {
      removeToast(id);
    }, duration);
  }

  function removeToast(id: string) {
    const index = toasts.value.findIndex(t => t.id === id);
    if (index !== -1) {
      toasts.value.splice(index, 1);
    }
  }

  function reset() {
    originalXml.value = null;
    fileName.value = '';
    parsedScore.value = null;
    patterns.value = [];
    fingeringSolution.value = null;
    placements.value = [];
    error.value = null;
  }

  function delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  return {
    // State
    originalXml,
    fileName,
    parsedScore,
    patterns,
    fingeringSolution,
    placements,
    isProcessing,
    processingStep,
    error,
    darkMode,
    colorMode,
    difficultyLevel,
    toasts,
    
    // Computed
    hasScore,
    hasFingering,
    noteCount,
    measureCount,
    rhNotes,
    lhNotes,
    patternSummary,
    
    // Actions
    uploadFile,
    generateFingering,
    updateNoteFingering,
    downloadMusicXML,
    toggleDarkMode,
    toggleColorMode,
    setDifficultyLevel,
    addToast,
    removeToast,
    reset
  };
});
