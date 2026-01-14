import type { 
  Note, PatternSegment, PatternType, Finger, Hand,
  FingeringState, FingeringSolution, CostResult 
} from '@/types';

/**
 * Layer 2: Fingering Optimization via Rule-based Dynamic Programming
 * 
 * CORRECTED based on validation against 27 real pieces (9186 fingerings)
 * Key principles from validation:
 * 1. Five-finger position is fundamental - map pitch position to finger
 * 2. Minimize hand position changes
 * 3. Use natural finger-to-pitch correspondence
 * 4. Thumb crossing only when necessary for scales
 */
export class FingeringPlanner {
  private difficultyLevel: 'beginner' | 'intermediate' | 'advanced' = 'intermediate';
  
  // Natural finger spans (in semitones)
  private readonly naturalSpans: Record<string, number> = {
    '1-2': 2, '2-3': 2, '3-4': 1, '4-5': 2,
    '1-3': 4, '2-4': 3, '3-5': 3,
    '1-4': 5, '2-5': 5,
    '1-5': 8
  };

  // Validated transition frequencies from real music
  private readonly transitionWeights: Record<string, number> = {
    '1->2': 900, '2->1': 936, '2->3': 405, '3->2': 528,
    '3->4': 420, '4->3': 409, '4->5': 590, '5->4': 561,
    '1->3': 451, '3->1': 489, '1->4': 301, '4->1': 286,
    '1->5': 356, '5->1': 286, '2->4': 150, '4->2': 149,
    '2->5': 334, '5->2': 248, '3->5': 271, '5->3': 448
  };

  setDifficultyLevel(level: 'beginner' | 'intermediate' | 'advanced') {
    this.difficultyLevel = level;
  }

  planFingering(notes: Note[], patterns: PatternSegment[]): FingeringSolution {
    if (notes.length === 0) {
      return { fingering: [], totalCost: 0, path: [], explanations: [] };
    }

    const rhNotes = notes.filter(n => n.hand === 'RH');
    const lhNotes = notes.filter(n => n.hand === 'LH');
    
    const rhSolution = this.planHandFingering(rhNotes, patterns, 'RH');
    const lhSolution = this.planHandFingering(lhNotes, patterns, 'LH');

    // Merge solutions
    const fingering: Finger[] = new Array(notes.length);
    const explanations: string[] = new Array(notes.length);
    
    let rhIdx = 0, lhIdx = 0;
    notes.forEach((note, i) => {
      if (note.hand === 'RH') {
        fingering[i] = rhSolution.fingering[rhIdx] || 3;
        explanations[i] = rhSolution.explanations[rhIdx] || '';
        rhIdx++;
      } else {
        fingering[i] = lhSolution.fingering[lhIdx] || 3;
        explanations[i] = lhSolution.explanations[lhIdx] || '';
        lhIdx++;
      }
    });
    
    return {
      fingering,
      totalCost: rhSolution.totalCost + lhSolution.totalCost,
      path: [...rhSolution.path, ...lhSolution.path],
      explanations
    };
  }

  private planHandFingering(notes: Note[], patterns: PatternSegment[], hand: Hand): FingeringSolution {
    if (notes.length === 0) {
      return { fingering: [], totalCost: 0, path: [], explanations: [] };
    }

    if (notes.length > 64) {
      return this.chunkedOptimization(notes, patterns, hand);
    }

    return this.dpOptimization(notes, patterns, hand);
  }

  /**
   * Core DP optimization with corrected cost function
   */
  private dpOptimization(notes: Note[], patterns: PatternSegment[], hand: Hand): FingeringSolution {
    const n = notes.length;
    const fingers: Finger[] = [1, 2, 3, 4, 5];
    
    // Analyze hand position context
    const handPositions = this.analyzeHandPositions(notes, hand);
    
    const dp: Map<string, { cost: number; parent: string | null; reasons: string[] }>[] = [];
    
    // Initialize first note
    dp[0] = new Map();
    const firstPos = handPositions[0];
    for (const finger of fingers) {
      const cost = this.computeInitialCost(finger, notes[0], hand, firstPos);
      dp[0].set(finger.toString(), { cost: cost.cost, parent: null, reasons: cost.reasons });
    }

    // Forward pass
    for (let i = 1; i < n; i++) {
      dp[i] = new Map();
      const pos = handPositions[i];
      const patternContext = this.getPatternContext(notes, i, patterns);
      
      for (const toFinger of fingers) {
        let minCost = Infinity;
        let bestParent: string | null = null;
        let bestReasons: string[] = [];
        
        for (const fromFinger of fingers) {
          const prevState = dp[i - 1].get(fromFinger.toString());
          if (!prevState) continue;
          
          const transitionCost = this.computeTransitionCost(
            notes[i - 1], fromFinger as Finger,
            notes[i], toFinger,
            patternContext, hand, pos
          );
          
          if (transitionCost.cost > 300) continue;
          
          const totalCost = prevState.cost + transitionCost.cost;
          
          if (totalCost < minCost) {
            minCost = totalCost;
            bestParent = `${i - 1}-${fromFinger}`;
            bestReasons = transitionCost.reasons;
          }
        }
        
        if (minCost < Infinity) {
          dp[i].set(toFinger.toString(), { cost: minCost, parent: bestParent, reasons: bestReasons });
        }
      }
    }
    
    return this.backtrack(dp, notes, hand);
  }

  /**
   * Analyze hand positions - KEY for correct fingering
   * Determines the "anchor" pitch for each segment
   * IMPROVED: Better detection of scale patterns that need thumb crossing
   */
  private analyzeHandPositions(notes: Note[], hand: Hand): { anchorPitch: number; inPosition: boolean; isScale: boolean }[] {
    const positions: { anchorPitch: number; inPosition: boolean; isScale: boolean }[] = [];
    
    if (notes.length === 0) return positions;
    
    // First, detect if this is a scale pattern (consecutive stepwise motion)
    const isScalePattern = this.detectScalePattern(notes);
    
    if (isScalePattern) {
      // For scales, we don't use fixed hand position - we use thumb crossing
      for (let i = 0; i < notes.length; i++) {
        positions[i] = { anchorPitch: notes[0].pitch, inPosition: false, isScale: true };
      }
      return positions;
    }
    
    // For non-scale patterns, find stable hand positions
    let segmentStart = 0;
    let minPitch = notes[0].pitch;
    let maxPitch = notes[0].pitch;
    
    for (let i = 0; i < notes.length; i++) {
      const pitch = notes[i].pitch;
      const newMin = Math.min(minPitch, pitch);
      const newMax = Math.max(maxPitch, pitch);
      
      // If range exceeds comfortable span (5th = 7 semitones), start new segment
      if (newMax - newMin > 7) {
        const anchorPitch = hand === 'RH' ? minPitch : maxPitch;
        for (let j = segmentStart; j < i; j++) {
          positions[j] = { anchorPitch, inPosition: true, isScale: false };
        }
        segmentStart = i;
        minPitch = pitch;
        maxPitch = pitch;
      } else {
        minPitch = newMin;
        maxPitch = newMax;
      }
    }
    
    const anchorPitch = hand === 'RH' ? minPitch : maxPitch;
    for (let j = segmentStart; j < notes.length; j++) {
      positions[j] = { anchorPitch, inPosition: true, isScale: false };
    }
    
    return positions;
  }

  /**
   * Detect if notes form a scale pattern (consecutive stepwise motion)
   */
  private detectScalePattern(notes: Note[]): boolean {
    if (notes.length < 8) return false; // Need at least 8 notes for a scale
    
    let stepwiseCount = 0;
    let sameDirection = 0;
    let lastDirection = 0;
    
    for (let i = 1; i < notes.length; i++) {
      const interval = notes[i].pitch - notes[i - 1].pitch;
      const absInterval = Math.abs(interval);
      
      if (absInterval === 1 || absInterval === 2) {
        stepwiseCount++;
        const direction = Math.sign(interval);
        if (direction === lastDirection && direction !== 0) {
          sameDirection++;
        }
        lastDirection = direction;
      }
    }
    
    const stepwiseRatio = stepwiseCount / (notes.length - 1);
    const directionRatio = notes.length > 2 ? sameDirection / (notes.length - 2) : 0;
    
    // More strict: need >80% stepwise AND >60% same direction
    // Also check total range - scales typically span more than an octave
    const pitches = notes.map(n => n.pitch);
    const range = Math.max(...pitches) - Math.min(...pitches);
    
    return stepwiseRatio > 0.8 && directionRatio > 0.6 && range > 7;
  }

  private chunkedOptimization(notes: Note[], patterns: PatternSegment[], hand: Hand): FingeringSolution {
    const chunkSize = 32;
    const overlapSize = 4;
    const chunks: Note[][] = [];
    
    for (let i = 0; i < notes.length; i += chunkSize - overlapSize) {
      const end = Math.min(i + chunkSize, notes.length);
      chunks.push(notes.slice(i, end));
    }
    
    const chunkSolutions = chunks.map(chunk => this.dpOptimization(chunk, patterns, hand));
    
    const fingering: Finger[] = [];
    const explanations: string[] = [];
    let totalCost = 0;
    
    for (let i = 0; i < chunkSolutions.length; i++) {
      const sol = chunkSolutions[i];
      if (i === 0) {
        fingering.push(...sol.fingering);
        explanations.push(...sol.explanations);
      } else {
        fingering.push(...sol.fingering.slice(overlapSize));
        explanations.push(...sol.explanations.slice(overlapSize));
      }
      totalCost += sol.totalCost;
    }
    
    return { fingering, totalCost, path: [], explanations };
  }

  private backtrack(
    dp: Map<string, { cost: number; parent: string | null; reasons: string[] }>[],
    notes: Note[],
    hand: Hand
  ): FingeringSolution {
    const n = notes.length;
    
    let minFinalCost = Infinity;
    let bestFinalFinger: Finger = 3;
    
    for (const [finger, state] of dp[n - 1].entries()) {
      if (state.cost < minFinalCost) {
        minFinalCost = state.cost;
        bestFinalFinger = parseInt(finger) as Finger;
      }
    }
    
    const fingering: Finger[] = new Array(n);
    const explanations: string[] = new Array(n);
    const path: FingeringState[] = [];
    
    let currentFinger = bestFinalFinger;
    
    for (let i = n - 1; i >= 0; i--) {
      fingering[i] = currentFinger;
      const state = dp[i].get(currentFinger.toString());
      explanations[i] = state?.reasons.join('; ') || '';
      
      path.unshift({
        noteIndex: i,
        finger: currentFinger,
        hand,
        handPosition: notes[i].pitch,
        cost: state?.cost || 0,
        parent: null,
        reasons: state?.reasons || []
      });
      
      if (state?.parent) {
        const [, prevFinger] = state.parent.split('-');
        currentFinger = parseInt(prevFinger) as Finger;
      }
    }
    
    return { fingering, totalCost: minFinalCost, path, explanations };
  }

  /**
   * Initial cost - CORRECTED based on validation
   * Key insight: First note should use finger based on position in hand span
   */
  private computeInitialCost(
    finger: Finger, 
    note: Note, 
    hand: Hand,
    pos: { anchorPitch: number; inPosition: boolean }
  ): CostResult {
    let cost = 0;
    const reasons: string[] = [];
    
    // Calculate expected finger based on position relative to anchor
    const offset = note.pitch - pos.anchorPitch;
    let expectedFinger: Finger;
    
    if (hand === 'RH') {
      // RH: thumb on lowest, pinky on highest
      if (offset <= 0) expectedFinger = 1;
      else if (offset <= 2) expectedFinger = 2;
      else if (offset <= 4) expectedFinger = 3;
      else if (offset <= 5) expectedFinger = 4;
      else expectedFinger = 5;
    } else {
      // LH: pinky on lowest, thumb on highest
      if (offset >= 0) expectedFinger = 1;
      else if (offset >= -2) expectedFinger = 2;
      else if (offset >= -4) expectedFinger = 3;
      else if (offset >= -5) expectedFinger = 4;
      else expectedFinger = 5;
    }
    
    // Strong reward for matching expected finger
    if (finger === expectedFinger) {
      cost -= 30;
      reasons.push('Matches hand position');
    } else {
      // Penalty based on distance from expected
      const diff = Math.abs(finger - expectedFinger);
      cost += diff * 15;
      reasons.push(`${diff} fingers from expected`);
    }
    
    // Black key adjustment
    if (this.isBlackKey(note.pitch)) {
      if (finger === 1 || finger === 5) {
        cost += 20;
        reasons.push('Short finger on black key');
      } else {
        cost -= 5;
        reasons.push('Long finger on black key');
      }
    }
    
    return { cost, reasons };
  }

  /**
   * Transition cost - CORRECTED based on validation
   * Key insight: Finger should follow pitch direction naturally
   */
  private computeTransitionCost(
    prevNote: Note, prevFinger: Finger,
    currNote: Note, currFinger: Finger,
    patternContext: PatternType,
    hand: Hand,
    pos: { anchorPitch: number; inPosition: boolean; isScale: boolean }
  ): CostResult {
    let cost = 0;
    const reasons: string[] = [];
    
    const interval = currNote.pitch - prevNote.pitch;
    const absInterval = Math.abs(interval);
    const ascending = interval > 0;
    const isStepwise = absInterval === 1 || absInterval === 2;
    
    // SCALE MODE: Use thumb crossing logic
    if (pos.isScale || patternContext === 'SCALE') {
      return this.computeScaleTransitionCost(prevFinger, currFinger, ascending, hand, absInterval);
    }
    
    // POSITION MODE: Use five-finger position mapping
    const offset = currNote.pitch - pos.anchorPitch;
    let expectedFinger: Finger;
    
    if (hand === 'RH') {
      if (offset <= 0) expectedFinger = 1;
      else if (offset <= 2) expectedFinger = 2;
      else if (offset <= 4) expectedFinger = 3;
      else if (offset <= 5) expectedFinger = 4;
      else expectedFinger = 5;
    } else {
      if (offset >= 0) expectedFinger = 1;
      else if (offset >= -2) expectedFinger = 2;
      else if (offset >= -4) expectedFinger = 3;
      else if (offset >= -5) expectedFinger = 4;
      else expectedFinger = 5;
    }
    
    // Strong reward for matching expected finger in position
    if (pos.inPosition && currFinger === expectedFinger) {
      cost -= 40;
      reasons.push('Correct finger for position');
    } else if (pos.inPosition) {
      const diff = Math.abs(currFinger - expectedFinger);
      cost += diff * 20;
      reasons.push(`Wrong finger for position`);
    }
    
    // Natural finger progression
    const fingerDiff = currFinger - prevFinger;
    
    if (hand === 'RH') {
      if (ascending && fingerDiff > 0) {
        cost -= 15;
        reasons.push('Natural RH ascending');
      } else if (!ascending && interval < 0 && fingerDiff < 0) {
        cost -= 15;
        reasons.push('Natural RH descending');
      } else if (interval !== 0 && fingerDiff !== 0) {
        if (prevFinger !== 1 && currFinger !== 1) {
          cost += 50;
          reasons.push('Unnatural finger crossing');
        }
      }
    } else {
      if (ascending && fingerDiff < 0) {
        cost -= 15;
        reasons.push('Natural LH ascending');
      } else if (!ascending && interval < 0 && fingerDiff > 0) {
        cost -= 15;
        reasons.push('Natural LH descending');
      } else if (interval !== 0 && fingerDiff !== 0) {
        if (prevFinger !== 1 && currFinger !== 1) {
          cost += 50;
          reasons.push('Unnatural finger crossing');
        }
      }
    }

    // RULE 3: Span constraint
    const naturalSpan = this.getNaturalSpan(prevFinger, currFinger);
    const overStretch = absInterval - naturalSpan;
    
    if (overStretch > 4) {
      cost += overStretch * 12;
      reasons.push('Over-stretch');
    } else if (overStretch > 2) {
      cost += overStretch * 6;
    }
    
    // RULE 4: Same finger on different pitch
    if (currFinger === prevFinger && interval !== 0) {
      cost += absInterval * 8;
      reasons.push('Same finger leap');
    }
    
    // RULE 5: Repeated note - use different finger
    if (interval === 0 && currFinger === prevFinger) {
      cost += 30;
      reasons.push('Same finger on repeated note');
    } else if (interval === 0 && currFinger !== prevFinger) {
      cost -= 10;
      reasons.push('Good finger change on repeat');
    }
    
    // RULE 6: Validated transition bonus
    const transKey = `${prevFinger}->${currFinger}`;
    const transWeight = this.transitionWeights[transKey];
    if (transWeight) {
      if (transWeight > 500) {
        cost -= 10;
      } else if (transWeight > 300) {
        cost -= 6;
      } else if (transWeight > 100) {
        cost -= 3;
      }
    }
    
    // RULE 7: Black key preference for long fingers
    if (this.isBlackKey(currNote.pitch)) {
      if (currFinger === 1 || currFinger === 5) {
        cost += 25;
        reasons.push('Short finger on black key');
      } else {
        cost -= 5;
      }
    }
    
    // RULE 8: Thumb crossing for scales
    const isScaleContext = patternContext === 'SCALE' as PatternType;
    if (isScaleContext) {
      if (hand === 'RH') {
        if (ascending && prevFinger === 3 && currFinger === 1) {
          cost -= 20;
          reasons.push('Good thumb under (3->1)');
        } else if (!ascending && prevFinger === 1 && currFinger === 3) {
          cost -= 20;
          reasons.push('Good finger over (1->3)');
        }
      } else {
        if (!ascending && prevFinger === 3 && currFinger === 1) {
          cost -= 20;
          reasons.push('Good thumb under (3->1)');
        } else if (ascending && prevFinger === 1 && currFinger === 3) {
          cost -= 20;
          reasons.push('Good finger over (1->3)');
        }
      }
    }
    
    // Apply difficulty adjustment
    cost = this.applyDifficultyAdjustment(cost, patternContext);
    
    return { cost, reasons };
  }

  /**
   * Special cost function for scale patterns
   * Uses standard scale fingering: 1-2-3-1-2-3-4-5 (ascending RH)
   */
  private computeScaleTransitionCost(
    prevFinger: Finger, currFinger: Finger,
    ascending: boolean, hand: Hand, interval: number
  ): CostResult {
    let cost = 0;
    const reasons: string[] = [];
    
    // Standard scale fingering patterns
    // RH ascending: 1-2-3-1-2-3-4-5 (thumb under after 3)
    // RH descending: 5-4-3-2-1-3-2-1 (finger over after 1)
    // LH ascending: 5-4-3-2-1-3-2-1
    // LH descending: 1-2-3-1-2-3-4-5
    
    const rhAscending = hand === 'RH' && ascending;
    const rhDescending = hand === 'RH' && !ascending;
    const lhAscending = hand === 'LH' && ascending;
    const lhDescending = hand === 'LH' && !ascending;
    
    // Good scale transitions
    const goodTransitions: [number, number][] = [];
    
    if (rhAscending || lhDescending) {
      // 1-2-3-1-2-3-4-5 pattern
      goodTransitions.push([1, 2], [2, 3], [3, 1], [3, 4], [4, 5]);
    }
    if (rhDescending || lhAscending) {
      // 5-4-3-2-1-3-2-1 pattern
      goodTransitions.push([5, 4], [4, 3], [3, 2], [2, 1], [1, 3], [1, 2]);
    }
    
    // Check if current transition is good
    const isGoodTransition = goodTransitions.some(
      ([from, to]) => from === prevFinger && to === currFinger
    );
    
    if (isGoodTransition) {
      cost -= 30;
      reasons.push('Standard scale fingering');
    } else {
      // Check for thumb crossing
      if (prevFinger === 3 && currFinger === 1) {
        cost -= 20;
        reasons.push('Thumb under (3->1)');
      } else if (prevFinger === 4 && currFinger === 1) {
        cost -= 15;
        reasons.push('Thumb under (4->1)');
      } else if (prevFinger === 1 && currFinger === 3) {
        cost -= 20;
        reasons.push('Finger over (1->3)');
      } else if (prevFinger === 1 && currFinger === 4) {
        cost -= 10;
        reasons.push('Finger over (1->4)');
      } else {
        // Non-standard transition
        cost += 20;
        reasons.push('Non-standard scale transition');
      }
    }
    
    // Penalize same finger
    if (currFinger === prevFinger) {
      cost += 40;
      reasons.push('Same finger in scale');
    }
    
    // Span check
    const naturalSpan = this.getNaturalSpan(prevFinger, currFinger);
    if (interval > naturalSpan + 2) {
      cost += (interval - naturalSpan) * 10;
      reasons.push('Over-stretch in scale');
    }
    
    return { cost, reasons };
  }

  private getPatternContext(notes: Note[], index: number, patterns: PatternSegment[]): PatternType {
    const note = notes[index];
    
    for (const pattern of patterns) {
      if (note.measureNumber >= pattern.startIndex && note.measureNumber <= pattern.endIndex) {
        return pattern.patternType;
      }
    }
    
    for (const pattern of patterns) {
      if (index >= pattern.startIndex && index <= pattern.endIndex) {
        return pattern.patternType;
      }
    }
    
    return 'UNKNOWN';
  }

  private getNaturalSpan(finger1: Finger, finger2: Finger): number {
    const key = `${Math.min(finger1, finger2)}-${Math.max(finger1, finger2)}`;
    return this.naturalSpans[key] || 0;
  }

  private isBlackKey(pitch: number): boolean {
    const pitchClass = pitch % 12;
    return [1, 3, 6, 8, 10].includes(pitchClass);
  }

  private applyDifficultyAdjustment(cost: number, patternContext: PatternType): number {
    switch (this.difficultyLevel) {
      case 'beginner':
        if (patternContext === 'POLYPHONIC' || patternContext === 'ORNAMENTED') {
          return cost * 1.3;
        }
        return cost * 1.1;
      case 'advanced':
        return cost * 0.9;
      default:
        return cost;
    }
  }
}

export const fingeringPlanner = new FingeringPlanner();
