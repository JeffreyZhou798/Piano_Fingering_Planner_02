import type { Note, MeasureInfo, ParsedScore, Hand, Finger } from '@/types';
import JSZip from 'jszip';

export class MusicXMLParser {
  private xmlDoc: Document | null = null;
  private divisions: number = 1;
  private currentMeasure: number = 0;
  private currentBeat: number = 0;
  private noteIdCounter: number = 0;

  async parseFile(file: File): Promise<ParsedScore> {
    const content = await this.readFile(file);
    return this.parseXML(content);
  }

  private async readFile(file: File): Promise<string> {
    const fileName = file.name.toLowerCase();
    
    if (fileName.endsWith('.mxl')) {
      return this.extractMXL(file);
    } else if (fileName.endsWith('.musicxml') || fileName.endsWith('.xml')) {
      return file.text();
    } else {
      throw new Error('Unsupported file format. Please upload .mxl, .musicxml, or .xml files.');
    }
  }

  private async extractMXL(file: File): Promise<string> {
    const zip = new JSZip();
    const contents = await zip.loadAsync(file);
    
    // Find container.xml to get the root file
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
    
    // Fallback: find any .xml file that's not in META-INF
    for (const [path, zipEntry] of Object.entries(contents.files)) {
      if (path.endsWith('.xml') && !path.startsWith('META-INF') && !zipEntry.dir) {
        return zipEntry.async('string');
      }
    }
    
    throw new Error('Could not find MusicXML content in the MXL file.');
  }

  parseXML(xmlString: string): ParsedScore {
    const parser = new DOMParser();
    this.xmlDoc = parser.parseFromString(xmlString, 'text/xml');
    
    const parseError = this.xmlDoc.querySelector('parsererror');
    if (parseError) {
      throw new Error('Invalid XML format: ' + parseError.textContent);
    }

    const title = this.extractTitle();
    const composer = this.extractComposer();
    const { notes, measures } = this.extractNotesAndMeasures();
    
    return {
      title,
      composer,
      notes,
      measures,
      divisions: this.divisions,
      partCount: this.xmlDoc.querySelectorAll('part').length
    };
  }

  private extractTitle(): string {
    const workTitle = this.xmlDoc?.querySelector('work-title');
    if (workTitle?.textContent) return workTitle.textContent;
    
    const movementTitle = this.xmlDoc?.querySelector('movement-title');
    if (movementTitle?.textContent) return movementTitle.textContent;
    
    return 'Untitled';
  }

  private extractComposer(): string {
    const creator = this.xmlDoc?.querySelector('creator[type="composer"]');
    if (creator?.textContent) return creator.textContent;
    
    const identification = this.xmlDoc?.querySelector('identification creator');
    if (identification?.textContent) return identification.textContent;
    
    return 'Unknown';
  }

  private extractNotesAndMeasures(): { notes: Note[]; measures: MeasureInfo[] } {
    const notes: Note[] = [];
    const measures: MeasureInfo[] = [];
    
    const parts = this.xmlDoc?.querySelectorAll('part');
    if (!parts || parts.length === 0) {
      throw new Error('No parts found in the score.');
    }

    // Process first part (piano part for grand staff)
    const part = parts[0];
    const measureElements = part.querySelectorAll('measure');
    
    let currentKeySignature = 0;
    let currentTimeSignature: [number, number] = [4, 4];
    let currentClefs: { staff: number; sign: string; line: number }[] = [];
    
    measureElements.forEach((measureEl, measureIndex) => {
      this.currentMeasure = measureIndex + 1;
      this.currentBeat = 0;
      
      // Extract attributes
      const attributes = measureEl.querySelector('attributes');
      if (attributes) {
        const divisionsEl = attributes.querySelector('divisions');
        if (divisionsEl?.textContent) {
          this.divisions = parseInt(divisionsEl.textContent, 10);
        }
        
        const keyEl = attributes.querySelector('key fifths');
        if (keyEl?.textContent) {
          currentKeySignature = parseInt(keyEl.textContent, 10);
        }
        
        const timeEl = attributes.querySelector('time');
        if (timeEl) {
          const beats = timeEl.querySelector('beats')?.textContent;
          const beatType = timeEl.querySelector('beat-type')?.textContent;
          if (beats && beatType) {
            currentTimeSignature = [parseInt(beats, 10), parseInt(beatType, 10)];
          }
        }
        
        const clefEls = attributes.querySelectorAll('clef');
        if (clefEls.length > 0) {
          currentClefs = [];
          clefEls.forEach(clefEl => {
            const staffNum = parseInt(clefEl.getAttribute('number') || '1', 10);
            const sign = clefEl.querySelector('sign')?.textContent || 'G';
            const line = parseInt(clefEl.querySelector('line')?.textContent || '2', 10);
            currentClefs.push({ staff: staffNum, sign, line });
          });
        }
      }
      
      measures.push({
        number: this.currentMeasure,
        divisions: this.divisions,
        timeSignature: currentTimeSignature,
        keySignature: currentKeySignature,
        clefs: [...currentClefs]
      });
      
      // Extract notes
      const noteElements = measureEl.querySelectorAll('note');
      let xmlIndex = 0;
      
      noteElements.forEach(noteEl => {
        const note = this.parseNoteElement(noteEl, xmlIndex, currentKeySignature);
        if (note && !note.isRest) {
          notes.push(note);
        }
        
        // Update beat position
        if (!noteEl.querySelector('chord')) {
          const duration = parseInt(noteEl.querySelector('duration')?.textContent || '0', 10);
          this.currentBeat += duration / this.divisions;
        }
        
        xmlIndex++;
      });
    });
    
    return { notes, measures };
  }

  private parseNoteElement(noteEl: Element, xmlIndex: number, keySignature: number): Note | null {
    const isRest = noteEl.querySelector('rest') !== null;
    const isChord = noteEl.querySelector('chord') !== null;
    const isGrace = noteEl.querySelector('grace') !== null;
    
    // Get pitch info
    const pitchEl = noteEl.querySelector('pitch');
    let pitch = 60; // Default to middle C
    let step = 'C';
    let octave = 4;
    let alter = 0;
    
    if (pitchEl) {
      step = pitchEl.querySelector('step')?.textContent || 'C';
      octave = parseInt(pitchEl.querySelector('octave')?.textContent || '4', 10);
      alter = parseInt(pitchEl.querySelector('alter')?.textContent || '0', 10);
      pitch = this.stepToMidi(step, octave, alter);
    }
    
    // Get duration and type
    const duration = parseInt(noteEl.querySelector('duration')?.textContent || '1', 10);
    const type = noteEl.querySelector('type')?.textContent || 'quarter';
    
    // Get voice and staff
    const voice = parseInt(noteEl.querySelector('voice')?.textContent || '1', 10);
    const staff = parseInt(noteEl.querySelector('staff')?.textContent || '1', 10);
    
    // Determine hand based on staff
    const hand: Hand = staff === 1 ? 'RH' : 'LH';
    
    // Get stem direction
    const stemEl = noteEl.querySelector('stem');
    const stem = stemEl?.textContent as 'up' | 'down' | undefined;
    
    // Check for articulations and ornaments
    const notations = noteEl.querySelector('notations');
    const articulations = notations?.querySelector('articulations');
    const ornaments = notations?.querySelector('ornaments');
    const technical = notations?.querySelector('technical');
    
    // Slurs
    const slurs = notations?.querySelectorAll('slur') || [];
    let hasSlur = false;
    let slurStart = false;
    let slurStop = false;
    slurs.forEach(slur => {
      hasSlur = true;
      if (slur.getAttribute('type') === 'start') slurStart = true;
      if (slur.getAttribute('type') === 'stop') slurStop = true;
    });
    
    // Ties
    const ties = noteEl.querySelectorAll('tie');
    let hasTie = false;
    let tieStart = false;
    let tieStop = false;
    ties.forEach(tie => {
      hasTie = true;
      if (tie.getAttribute('type') === 'start') tieStart = true;
      if (tie.getAttribute('type') === 'stop') tieStop = true;
    });
    
    // Articulations
    const hasAccent = articulations?.querySelector('accent') !== null || 
                      articulations?.querySelector('strong-accent') !== null;
    const hasStaccato = articulations?.querySelector('staccato') !== null;
    
    // Ornaments
    const hasTrill = ornaments?.querySelector('trill-mark') !== null;
    const hasMordent = ornaments?.querySelector('mordent') !== null || 
                       ornaments?.querySelector('inverted-mordent') !== null;
    const hasTurn = ornaments?.querySelector('turn') !== null || 
                   ornaments?.querySelector('inverted-turn') !== null;
    
    // Existing fingering
    const fingeringEl = technical?.querySelector('fingering');
    let fingering: Finger | undefined;
    if (fingeringEl?.textContent) {
      const f = parseInt(fingeringEl.textContent, 10);
      if (f >= 1 && f <= 5) {
        fingering = f as Finger;
      }
    }
    
    // Dynamics
    const directionEl = noteEl.parentElement?.querySelector('direction dynamics');
    const dynamic = directionEl?.firstElementChild?.tagName;
    
    return {
      id: `note-${this.noteIdCounter++}`,
      pitch,
      step,
      octave,
      alter,
      duration,
      type,
      voice,
      staff,
      hand,
      measureNumber: this.currentMeasure,
      beat: this.currentBeat,
      isChord,
      isGrace,
      isRest,
      hasSlur,
      slurStart,
      slurStop,
      hasTie,
      tieStart,
      tieStop,
      hasAccent,
      hasStaccato,
      hasTrill,
      hasMordent,
      hasTurn,
      dynamic,
      stem,
      fingering,
      xmlIndex
    };
  }

  private stepToMidi(step: string, octave: number, alter: number): number {
    const stepMap: Record<string, number> = {
      'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11
    };
    return (octave + 1) * 12 + stepMap[step] + alter;
  }

  // Write fingering back to MusicXML
  writeFingeringToXML(originalXml: string, notes: Note[]): string {
    const parser = new DOMParser();
    const doc = parser.parseFromString(originalXml, 'text/xml');
    
    const parts = doc.querySelectorAll('part');
    if (parts.length === 0) return originalXml;
    
    const part = parts[0];
    const measureElements = part.querySelectorAll('measure');
    
    // Create a map of notes by measure and xmlIndex
    const noteMap = new Map<string, Note>();
    notes.forEach(note => {
      const key = `${note.measureNumber}-${note.xmlIndex}`;
      noteMap.set(key, note);
    });
    
    measureElements.forEach((measureEl, measureIndex) => {
      const measureNum = measureIndex + 1;
      const noteElements = measureEl.querySelectorAll('note');
      
      noteElements.forEach((noteEl, xmlIndex) => {
        const key = `${measureNum}-${xmlIndex}`;
        const note = noteMap.get(key);
        
        if (note?.fingering && !noteEl.querySelector('rest')) {
          // Find or create notations element
          let notations = noteEl.querySelector('notations');
          if (!notations) {
            notations = doc.createElement('notations');
            noteEl.appendChild(notations);
          }
          
          // Find or create technical element
          let technical = notations.querySelector('technical');
          if (!technical) {
            technical = doc.createElement('technical');
            notations.appendChild(technical);
          }
          
          // Remove existing fingering
          const existingFingering = technical.querySelector('fingering');
          if (existingFingering) {
            technical.removeChild(existingFingering);
          }
          
          // Add new fingering
          const fingeringEl = doc.createElement('fingering');
          fingeringEl.textContent = note.fingering.toString();
          
          // Set placement based on hand/staff
          const placement = note.hand === 'RH' ? 'above' : 'below';
          fingeringEl.setAttribute('placement', placement);
          
          technical.appendChild(fingeringEl);
        }
      });
    });
    
    const serializer = new XMLSerializer();
    return serializer.serializeToString(doc);
  }
}

export const musicXMLParser = new MusicXMLParser();
