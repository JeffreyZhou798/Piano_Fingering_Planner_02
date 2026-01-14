// Pattern Types - 11 categories as per architecture
export type PrimaryPatternType = 
  | 'SCALE' 
  | 'ARPEGGIO' 
  | 'REPEATED' 
  | 'LEAP' 
  | 'CHORDAL' 
  | 'MELODIC' 
  | 'UNKNOWN';

export type SpecialPatternType = 
  | 'ALBERTI' 
  | 'ORNAMENTED' 
  | 'OSTINATO' 
  | 'POLYPHONIC';

export type PatternType = PrimaryPatternType | SpecialPatternType;

// Hand and Finger definitions
export type Hand = 'RH' | 'LH';
export type Finger = 1 | 2 | 3 | 4 | 5;

// Note representation
export interface Note {
  id: string;
  pitch: number; // MIDI pitch (0-127)
  step: string; // C, D, E, F, G, A, B
  octave: number;
  alter: number; // -1 flat, 0 natural, 1 sharp
  duration: number; // in divisions
  type: string; // whole, half, quarter, eighth, 16th
  voice: number;
  staff: number; // 1 = treble, 2 = bass
  hand: Hand;
  measureNumber: number;
  beat: number;
  isChord: boolean;
  isGrace: boolean;
  isRest: boolean;
  
  // Articulation marks
  hasSlur: boolean;
  slurStart: boolean;
  slurStop: boolean;
  hasTie: boolean;
  tieStart: boolean;
  tieStop: boolean;
  hasAccent: boolean;
  hasStaccato: boolean;
  
  // Ornaments
  hasTrill: boolean;
  hasMordent: boolean;
  hasTurn: boolean;
  
  // Dynamics
  dynamic?: string;
  
  // Stem direction
  stem?: 'up' | 'down';
  
  // Fingering (if already present or assigned)
  fingering?: Finger;
  
  // Position in XML for writing back
  xmlIndex: number;
}

// Chord representation (multiple notes at same time)
export interface Chord {
  notes: Note[];
  duration: number;
  measureNumber: number;
  beat: number;
  hand: Hand;
}

// Pattern segment from Layer 1
export interface PatternSegment {
  startIndex: number;
  endIndex: number;
  patternType: PatternType;
  confidence: number;
  features: PatternFeatures;
}

export interface PatternFeatures {
  direction?: 'ascending' | 'descending' | 'bidirectional' | 'mixed';
  scaleType?: 'major' | 'minor' | 'chromatic' | 'pentatonic' | 'modal';
  chordType?: 'major' | 'minor' | 'diminished' | 'augmented' | 'dominant7' | 'major7' | 'complex';
  chordRoot?: number;
  inversion?: 0 | 1 | 2;
  repeatCount?: number;
  repeatType?: 'single' | 'alternating' | 'pattern';
  ornamentType?: 'trill' | 'mordent' | 'turn' | 'grace';
  intervalPattern?: number[];
  contour?: 'jagged' | 'arch' | 'valley' | 'linear';
  style?: 'cantabile' | 'lyrical' | 'expressive' | 'neutral';
}

// Fingering state for DP
export interface FingeringState {
  noteIndex: number;
  finger: Finger;
  hand: Hand;
  handPosition: number;
  cost: number;
  parent: FingeringState | null;
  reasons: string[];
}

// Fingering solution
export interface FingeringSolution {
  fingering: Finger[];
  totalCost: number;
  path: FingeringState[];
  explanations: string[];
}

// Cost calculation result
export interface CostResult {
  cost: number;
  reasons: string[];
}

// Measure info
export interface MeasureInfo {
  number: number;
  divisions: number;
  timeSignature: [number, number];
  keySignature: number; // -7 to +7 (flats to sharps)
  clefs: { staff: number; sign: string; line: number }[];
}

// Parsed score
export interface ParsedScore {
  title: string;
  composer: string;
  notes: Note[];
  measures: MeasureInfo[];
  divisions: number;
  partCount: number;
}

// Fingering placement for visualization
export interface FingeringPlacement {
  noteIndex: number;
  finger: Finger;
  x: number;
  y: number;
  placement: 'above' | 'below';
  hand: Hand;
  style: {
    fontSize: string;
    fontWeight: string;
    color: string;
  };
}

// Application state
export interface AppState {
  originalXml: string | null;
  parsedScore: ParsedScore | null;
  patterns: PatternSegment[];
  fingeringSolution: FingeringSolution | null;
  placements: FingeringPlacement[];
  isProcessing: boolean;
  error: string | null;
  darkMode: boolean;
  colorMode: boolean;
  difficultyLevel: 'beginner' | 'intermediate' | 'advanced';
}

// Toast notification
export interface Toast {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  message: string;
  duration: number;
}
