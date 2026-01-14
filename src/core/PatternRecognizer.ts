import type { Note, PatternSegment, PatternType, PatternFeatures } from '@/types';

/**
 * Layer 1: Pattern Recognition via Decision Tree
 * Identifies 11 pattern types (7 primary + 4 special)
 */
export class PatternRecognizer {
  private readonly baseWindowSize = 8;
  private readonly overlapSize = 4;

  recognizePatterns(notes: Note[]): PatternSegment[] {
    if (notes.length === 0) return [];
    
    // Separate by hand
    const rhNotes = notes.filter(n => n.hand === 'RH');
    const lhNotes = notes.filter(n => n.hand === 'LH');
    
    const rhPatterns = this.recognizeHandPatterns(rhNotes, 'RH');
    const lhPatterns = this.recognizeHandPatterns(lhNotes, 'LH');
    
    // Merge and sort by start index
    return [...rhPatterns, ...lhPatterns].sort((a, b) => a.startIndex - b.startIndex);
  }

  private recognizeHandPatterns(notes: Note[], hand: string): PatternSegment[] {
    if (notes.length === 0) return [];
    
    const segments: PatternSegment[] = [];
    let i = 0;
    
    while (i < notes.length) {
      const windowSize = this.adaptWindowSize(notes, i);
      const windowEnd = Math.min(i + windowSize, notes.length);
      const window = notes.slice(i, windowEnd);
      
      if (window.length < 2) {
        i++;
        continue;
      }
      
      const features = this.extractFeatures(window);
      const result = this.classifyPattern(window, features);
      
      segments.push({
        startIndex: notes.indexOf(window[0]),
        endIndex: notes.indexOf(window[window.length - 1]),
        patternType: result.type,
        confidence: result.confidence,
        features: result.features
      });
      
      // Move forward, accounting for overlap
      const step = Math.max(1, Math.floor(windowSize / 2));
      i += step;
    }
    
    return this.postProcess(segments, notes);
  }

  private adaptWindowSize(notes: Note[], startIdx: number): number {
    // Calculate local density
    const lookAhead = Math.min(16, notes.length - startIdx);
    if (lookAhead < 4) return lookAhead;
    
    const window = notes.slice(startIdx, startIdx + lookAhead);
    const totalDuration = window.reduce((sum, n) => sum + n.duration, 0);
    const avgDuration = totalDuration / window.length;
    
    // Fast passages need larger windows
    if (avgDuration < 0.25) return 16;
    if (avgDuration < 0.5) return 12;
    if (avgDuration > 2) return 4;
    
    return this.baseWindowSize;
  }

  private extractFeatures(notes: Note[]): ExtractedFeatures {
    const pitches = notes.map(n => n.pitch);
    const intervals = this.computeIntervals(pitches);
    
    // Pitch features
    const pitchRange = Math.max(...pitches) - Math.min(...pitches);
    const pitchEntropy = this.computeEntropy(pitches);
    const ascendingCount = intervals.filter(i => i > 0).length;
    const descendingCount = intervals.filter(i => i < 0).length;
    const totalIntervals = intervals.length || 1;
    
    // Interval features
    const absIntervals = intervals.map(Math.abs);
    const maxInterval = absIntervals.length > 0 ? Math.max(...absIntervals) : 0;
    const avgInterval = absIntervals.length > 0 ? 
      absIntervals.reduce((a, b) => a + b, 0) / absIntervals.length : 0;
    const intervalVariance = this.computeVariance(absIntervals);
    
    // Stepwise motion (seconds)
    const stepwiseCount = absIntervals.filter(i => i <= 2).length;
    const stepwiseRatio = absIntervals.length > 0 ? stepwiseCount / absIntervals.length : 0;
    
    // Leap motion (> major third)
    const leapCount = absIntervals.filter(i => i > 4).length;
    const leapRatio = absIntervals.length > 0 ? leapCount / absIntervals.length : 0;
    
    // Direction changes
    const directionChanges = this.countDirectionChanges(intervals);
    
    // Simultaneity (chords)
    const simultaneity = this.computeSimultaneity(notes);
    
    // Rhythm features
    const durations = notes.map(n => n.duration);
    const durationVariance = this.computeVariance(durations);
    const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
    
    // Context features
    const hasSlur = notes.some(n => n.hasSlur);
    const hasOrnament = notes.some(n => n.hasTrill || n.hasMordent || n.hasTurn);
    const hasGrace = notes.some(n => n.isGrace);
    const staff = notes[0]?.staff || 1;
    
    return {
      pitches,
      intervals,
      pitchRange,
      pitchEntropy,
      ascendingRatio: ascendingCount / totalIntervals,
      descendingRatio: descendingCount / totalIntervals,
      maxInterval,
      avgInterval,
      intervalVariance,
      stepwiseRatio,
      leapRatio,
      directionChanges,
      simultaneity,
      avgSimultaneity: simultaneity.reduce((a, b) => a + b, 0) / (simultaneity.length || 1),
      maxSimultaneity: simultaneity.length > 0 ? Math.max(...simultaneity) : 1,
      durationVariance,
      avgDuration,
      hasSlur,
      hasOrnament,
      hasGrace,
      staff
    };
  }

  private classifyPattern(notes: Note[], features: ExtractedFeatures): ClassificationResult {
    // Priority 1: Special patterns
    const ornamented = this.detectOrnamented(notes, features);
    if (ornamented) return ornamented;
    
    const alberti = this.detectAlberti(notes, features);
    if (alberti) return alberti;
    
    const ostinato = this.detectOstinato(notes, features);
    if (ostinato) return ostinato;
    
    const polyphonic = this.detectPolyphonic(notes, features);
    if (polyphonic) return polyphonic;
    
    // Priority 2: Vertical density (chords)
    if (features.avgSimultaneity >= 2 || features.maxSimultaneity >= 3) {
      return this.classifyChordal(notes, features);
    }
    
    // Priority 3: Horizontal motion patterns
    // Scale detection
    if (features.stepwiseRatio >= 0.8) {
      const direction = this.determineDirection(features);
      if (direction !== 'mixed') {
        return this.classifyScale(notes, features, direction);
      }
    }
    
    // Arpeggio detection
    if (features.leapRatio >= 0.5 && this.belongsToChord(features.pitches)) {
      return this.classifyArpeggio(notes, features);
    }
    
    // Repeated notes detection
    if (features.pitchEntropy < 0.5) {
      const repeated = this.detectRepeated(notes, features);
      if (repeated) return repeated;
    }
    
    // Leap detection
    if (features.maxInterval > 4 && features.directionChanges > notes.length * 0.4) {
      return this.classifyLeap(notes, features);
    }
    
    // Melodic line (fallback with positive features)
    if (features.hasSlur || features.durationVariance > 0.3) {
      return this.classifyMelodic(notes, features);
    }
    
    // Unknown
    return {
      type: 'UNKNOWN',
      confidence: 0.5,
      features: {}
    };
  }

  // Special pattern detectors
  private detectOrnamented(notes: Note[], features: ExtractedFeatures): ClassificationResult | null {
    if (features.hasOrnament || features.hasGrace) {
      let ornamentType: 'trill' | 'mordent' | 'turn' | 'grace' = 'grace';
      
      if (notes.some(n => n.hasTrill)) ornamentType = 'trill';
      else if (notes.some(n => n.hasMordent)) ornamentType = 'mordent';
      else if (notes.some(n => n.hasTurn)) ornamentType = 'turn';
      
      return {
        type: 'ORNAMENTED',
        confidence: 1.0,
        features: { ornamentType }
      };
    }
    
    // Detect implicit ornaments (rapid alternation)
    if (features.avgDuration < 0.125 && features.maxInterval <= 2) {
      const isAlternating = this.isAlternatingPattern(features.intervals);
      if (isAlternating) {
        return {
          type: 'ORNAMENTED',
          confidence: 0.75,
          features: { ornamentType: 'trill' }
        };
      }
    }
    
    return null;
  }

  private detectAlberti(notes: Note[], features: ExtractedFeatures): ClassificationResult | null {
    // Must be in bass staff or low register
    if (features.staff !== 2 && features.pitches[0] >= 60) return null;
    
    // Check for low-high-mid-high pattern
    if (notes.length < 4) return null;
    
    let matchCount = 0;
    const groupSize = 4;
    
    for (let i = 0; i <= notes.length - groupSize; i += groupSize) {
      const group = notes.slice(i, i + groupSize);
      if (group.length !== 4) continue;
      
      const p = group.map(n => n.pitch);
      // Pattern: low-high-mid-high where low < mid < high
      if (p[0] < p[2] && p[2] < p[1] && Math.abs(p[1] - p[3]) <= 1) {
        matchCount++;
      }
    }
    
    const totalGroups = Math.floor(notes.length / groupSize);
    const matchRatio = totalGroups > 0 ? matchCount / totalGroups : 0;
    
    if (matchRatio >= 0.6) {
      return {
        type: 'ALBERTI',
        confidence: Math.min(0.95, 0.6 + matchRatio * 0.35),
        features: {
          repeatCount: matchCount
        }
      };
    }
    
    return null;
  }

  private detectOstinato(notes: Note[], features: ExtractedFeatures): ClassificationResult | null {
    // Try different pattern lengths
    for (let patternLen = 2; patternLen <= Math.min(8, Math.floor(notes.length / 3)); patternLen++) {
      const pattern = notes.slice(0, patternLen).map(n => n.pitch);
      let repeatCount = 0;
      
      for (let i = patternLen; i <= notes.length - patternLen; i += patternLen) {
        const candidate = notes.slice(i, i + patternLen).map(n => n.pitch);
        if (this.arraysEqual(pattern, candidate)) {
          repeatCount++;
        } else {
          break;
        }
      }
      
      if (repeatCount >= 2) {
        return {
          type: 'OSTINATO',
          confidence: Math.min(0.95, 0.7 + repeatCount * 0.05),
          features: {
            repeatCount: repeatCount + 1,
            intervalPattern: this.computeIntervals(pattern)
          }
        };
      }
    }
    
    return null;
  }

  private detectPolyphonic(notes: Note[], features: ExtractedFeatures): ClassificationResult | null {
    // Check for multiple voices
    const voices = new Set(notes.map(n => n.voice));
    if (voices.size < 2) return null;
    
    // Check rhythmic independence
    const voiceNotes = new Map<number, Note[]>();
    notes.forEach(n => {
      if (!voiceNotes.has(n.voice)) voiceNotes.set(n.voice, []);
      voiceNotes.get(n.voice)!.push(n);
    });
    
    // Calculate rhythm correlation between voices
    const voiceArrays = Array.from(voiceNotes.values());
    if (voiceArrays.length < 2) return null;
    
    const beats1 = new Set(voiceArrays[0].map(n => n.beat));
    const beats2 = new Set(voiceArrays[1].map(n => n.beat));
    
    let overlap = 0;
    beats1.forEach(b => { if (beats2.has(b)) overlap++; });
    
    const correlation = overlap / Math.max(beats1.size, beats2.size);
    
    if (correlation < 0.4) {
      return {
        type: 'POLYPHONIC',
        confidence: 0.8,
        features: {}
      };
    }
    
    return null;
  }

  private detectRepeated(notes: Note[], features: ExtractedFeatures): ClassificationResult | null {
    const pitches = features.pitches;
    
    // Single note repeat
    let maxRepeat = 1;
    let currentRepeat = 1;
    
    for (let i = 1; i < pitches.length; i++) {
      if (pitches[i] === pitches[i - 1]) {
        currentRepeat++;
        maxRepeat = Math.max(maxRepeat, currentRepeat);
      } else {
        currentRepeat = 1;
      }
    }
    
    if (maxRepeat >= 3) {
      return {
        type: 'REPEATED',
        confidence: Math.min(0.95, 0.7 + maxRepeat * 0.05),
        features: {
          repeatType: 'single',
          repeatCount: maxRepeat
        }
      };
    }
    
    // Alternating pattern
    if (pitches.length >= 4) {
      const isAlternating = pitches.every((p, i) => 
        i % 2 === 0 ? p === pitches[0] : p === pitches[1]
      );
      
      if (isAlternating && pitches[0] !== pitches[1]) {
        return {
          type: 'REPEATED',
          confidence: 0.85,
          features: {
            repeatType: 'alternating',
            repeatCount: Math.floor(pitches.length / 2)
          }
        };
      }
    }
    
    return null;
  }

  // Primary pattern classifiers
  private classifyChordal(notes: Note[], features: ExtractedFeatures): ClassificationResult {
    const chordAnalysis = this.analyzeChord(features.pitches);
    
    return {
      type: 'CHORDAL',
      confidence: 0.9,
      features: {
        chordType: chordAnalysis.type,
        chordRoot: chordAnalysis.root,
        inversion: chordAnalysis.inversion
      }
    };
  }

  private classifyScale(notes: Note[], features: ExtractedFeatures, direction: 'ascending' | 'descending' | 'bidirectional'): ClassificationResult {
    const scaleType = this.identifyScaleType(features.intervals);
    
    return {
      type: 'SCALE',
      confidence: 0.92,
      features: {
        direction,
        scaleType
      }
    };
  }

  private classifyArpeggio(notes: Note[], features: ExtractedFeatures): ClassificationResult {
    const chordAnalysis = this.analyzeChord(features.pitches);
    const direction = this.determineArpeggioDirection(features);
    
    return {
      type: 'ARPEGGIO',
      confidence: 0.88,
      features: {
        direction,
        chordType: chordAnalysis.type,
        chordRoot: chordAnalysis.root,
        inversion: chordAnalysis.inversion
      }
    };
  }

  private classifyLeap(notes: Note[], features: ExtractedFeatures): ClassificationResult {
    const contour = this.analyzeContour(features);
    
    return {
      type: 'LEAP',
      confidence: 0.8,
      features: {
        contour,
        direction: features.ascendingRatio > 0.6 ? 'ascending' : 
                   features.descendingRatio > 0.6 ? 'descending' : 'mixed'
      }
    };
  }

  private classifyMelodic(notes: Note[], features: ExtractedFeatures): ClassificationResult {
    let style: 'cantabile' | 'lyrical' | 'expressive' | 'neutral' = 'neutral';
    
    if (features.hasSlur && features.avgDuration > 1.0) {
      style = 'cantabile';
    } else if (features.durationVariance > 0.4) {
      style = 'expressive';
    } else if (features.hasSlur) {
      style = 'lyrical';
    }
    
    return {
      type: 'MELODIC',
      confidence: 0.7,
      features: { style }
    };
  }

  // Helper methods
  private computeIntervals(pitches: number[]): number[] {
    const intervals: number[] = [];
    for (let i = 1; i < pitches.length; i++) {
      intervals.push(pitches[i] - pitches[i - 1]);
    }
    return intervals;
  }

  private computeEntropy(values: number[]): number {
    const counts = new Map<number, number>();
    values.forEach(v => counts.set(v, (counts.get(v) || 0) + 1));
    
    let entropy = 0;
    const total = values.length;
    counts.forEach(count => {
      const p = count / total;
      if (p > 0) entropy -= p * Math.log2(p);
    });
    
    return entropy;
  }

  private computeVariance(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  }

  private countDirectionChanges(intervals: number[]): number {
    let changes = 0;
    for (let i = 1; i < intervals.length; i++) {
      if (Math.sign(intervals[i]) !== Math.sign(intervals[i - 1]) && 
          intervals[i] !== 0 && intervals[i - 1] !== 0) {
        changes++;
      }
    }
    return changes;
  }

  private computeSimultaneity(notes: Note[]): number[] {
    const beatMap = new Map<number, number>();
    notes.forEach(n => {
      const key = Math.round(n.beat * 100) / 100;
      beatMap.set(key, (beatMap.get(key) || 0) + 1);
    });
    return Array.from(beatMap.values());
  }

  private determineDirection(features: ExtractedFeatures): 'ascending' | 'descending' | 'bidirectional' | 'mixed' {
    if (features.ascendingRatio > 0.75) return 'ascending';
    if (features.descendingRatio > 0.75) return 'descending';
    if (features.ascendingRatio > 0.5 && features.descendingRatio > 0.3) return 'bidirectional';
    return 'mixed';
  }

  private determineArpeggioDirection(features: ExtractedFeatures): 'ascending' | 'descending' | 'bidirectional' | 'mixed' {
    const firstHalf = features.intervals.slice(0, Math.floor(features.intervals.length / 2));
    const secondHalf = features.intervals.slice(Math.floor(features.intervals.length / 2));
    
    const firstDir = firstHalf.reduce((a, b) => a + Math.sign(b), 0);
    const secondDir = secondHalf.reduce((a, b) => a + Math.sign(b), 0);
    
    if (firstDir > 0 && secondDir < 0) return 'bidirectional';
    if (firstDir > 0) return 'ascending';
    if (firstDir < 0) return 'descending';
    return 'mixed';
  }

  private belongsToChord(pitches: number[]): boolean {
    const pitchClasses = [...new Set(pitches.map(p => p % 12))].sort((a, b) => a - b);
    
    if (pitchClasses.length < 3 || pitchClasses.length > 4) return false;
    
    // Check for triad structure
    if (pitchClasses.length === 3) {
      const i1 = (pitchClasses[1] - pitchClasses[0] + 12) % 12;
      const i2 = (pitchClasses[2] - pitchClasses[1] + 12) % 12;
      
      // Major, minor, diminished, augmented
      if ((i1 === 4 && i2 === 3) || (i1 === 3 && i2 === 4) ||
          (i1 === 3 && i2 === 3) || (i1 === 4 && i2 === 4)) {
        return true;
      }
    }
    
    // Check for seventh chord
    if (pitchClasses.length === 4) {
      const intervals = [];
      for (let i = 1; i < pitchClasses.length; i++) {
        intervals.push((pitchClasses[i] - pitchClasses[i - 1] + 12) % 12);
      }
      // Dominant 7th, major 7th, minor 7th patterns
      if (intervals[0] >= 3 && intervals[0] <= 4 && 
          intervals[1] >= 3 && intervals[1] <= 4) {
        return true;
      }
    }
    
    return false;
  }

  private analyzeChord(pitches: number[]): { type: PatternFeatures['chordType']; root: number; inversion: 0 | 1 | 2 } {
    const pitchClasses = [...new Set(pitches.map(p => p % 12))].sort((a, b) => a - b);
    
    if (pitchClasses.length < 3) {
      return { type: 'complex', root: pitchClasses[0] || 0, inversion: 0 };
    }
    
    const i1 = (pitchClasses[1] - pitchClasses[0] + 12) % 12;
    const i2 = (pitchClasses[2] - pitchClasses[1] + 12) % 12;
    
    let type: PatternFeatures['chordType'] = 'complex';
    
    if (i1 === 4 && i2 === 3) type = 'major';
    else if (i1 === 3 && i2 === 4) type = 'minor';
    else if (i1 === 3 && i2 === 3) type = 'diminished';
    else if (i1 === 4 && i2 === 4) type = 'augmented';
    
    return { type, root: pitchClasses[0], inversion: 0 };
  }

  private identifyScaleType(intervals: number[]): PatternFeatures['scaleType'] {
    const absIntervals = intervals.map(Math.abs);
    
    // Chromatic: all semitones
    if (absIntervals.every(i => i === 1)) return 'chromatic';
    
    // Check for major/minor patterns
    const pattern = absIntervals.join(',');
    if (pattern.includes('2,2,1,2,2,2,1') || pattern.includes('2,2,1,2,2,2')) return 'major';
    if (pattern.includes('2,1,2,2,1,2,2') || pattern.includes('2,1,2,2,2,1')) return 'minor';
    
    // Pentatonic: only 2s and 3s
    if (absIntervals.every(i => i === 2 || i === 3)) return 'pentatonic';
    
    return 'modal';
  }

  private analyzeContour(features: ExtractedFeatures): 'jagged' | 'arch' | 'valley' | 'linear' {
    if (features.directionChanges > features.intervals.length * 0.5) return 'jagged';
    
    const midpoint = Math.floor(features.intervals.length / 2);
    const firstHalf = features.intervals.slice(0, midpoint);
    const secondHalf = features.intervals.slice(midpoint);
    
    const firstTrend = firstHalf.reduce((a, b) => a + b, 0);
    const secondTrend = secondHalf.reduce((a, b) => a + b, 0);
    
    if (firstTrend > 0 && secondTrend < 0) return 'arch';
    if (firstTrend < 0 && secondTrend > 0) return 'valley';
    
    return 'linear';
  }

  private isAlternatingPattern(intervals: number[]): boolean {
    if (intervals.length < 3) return false;
    
    for (let i = 0; i < intervals.length - 1; i++) {
      if (Math.abs(intervals[i]) > 2 || intervals[i] !== -intervals[i + 1]) {
        return false;
      }
    }
    return true;
  }

  private arraysEqual(a: number[], b: number[]): boolean {
    if (a.length !== b.length) return false;
    return a.every((v, i) => v === b[i]);
  }

  private postProcess(segments: PatternSegment[], notes: Note[]): PatternSegment[] {
    if (segments.length === 0) return segments;
    
    // Merge short segments
    const merged: PatternSegment[] = [];
    let current = segments[0];
    
    for (let i = 1; i < segments.length; i++) {
      const next = segments[i];
      const currentLength = current.endIndex - current.startIndex;
      
      // Merge if same type or current is very short
      if (current.patternType === next.patternType || currentLength < 3) {
        current = {
          ...current,
          endIndex: next.endIndex,
          confidence: Math.max(current.confidence, next.confidence)
        };
      } else {
        merged.push(current);
        current = next;
      }
    }
    merged.push(current);
    
    return merged;
  }
}

interface ExtractedFeatures {
  pitches: number[];
  intervals: number[];
  pitchRange: number;
  pitchEntropy: number;
  ascendingRatio: number;
  descendingRatio: number;
  maxInterval: number;
  avgInterval: number;
  intervalVariance: number;
  stepwiseRatio: number;
  leapRatio: number;
  directionChanges: number;
  simultaneity: number[];
  avgSimultaneity: number;
  maxSimultaneity: number;
  durationVariance: number;
  avgDuration: number;
  hasSlur: boolean;
  hasOrnament: boolean;
  hasGrace: boolean;
  staff: number;
}

interface ClassificationResult {
  type: PatternType;
  confidence: number;
  features: PatternFeatures;
}

export const patternRecognizer = new PatternRecognizer();
