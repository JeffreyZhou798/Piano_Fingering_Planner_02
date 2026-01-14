import type { Note, Finger, FingeringPlacement, Hand } from '@/types';

/**
 * Layer 3: Fingering Placement & Rendering
 * Handles visual positioning of fingering numbers
 */
export class FingeringRenderer {
  private colorMode: boolean = false;
  
  // Finger colors for educational mode
  private readonly fingerColors: Record<Finger, string> = {
    1: '#D32F2F', // Red - Thumb
    2: '#F57C00', // Orange - Index
    3: '#1976D2', // Blue - Middle
    4: '#388E3C', // Green - Ring
    5: '#7B1FA2'  // Purple - Pinky
  };

  setColorMode(enabled: boolean) {
    this.colorMode = enabled;
  }

  calculatePlacements(notes: Note[]): FingeringPlacement[] {
    const placements: FingeringPlacement[] = [];
    
    notes.forEach((note, index) => {
      if (note.fingering && !note.isRest) {
        const placement = this.calculateSinglePlacement(note, index);
        placements.push(placement);
      }
    });
    
    return this.resolveCollisions(placements);
  }

  private calculateSinglePlacement(note: Note, index: number): FingeringPlacement {
    // Determine vertical placement based on hand/staff
    const placement = this.determineVerticalPlacement(note);
    
    // Calculate base position
    const x = this.calculateXPosition(note, index);
    const y = this.calculateYPosition(note, placement);
    
    // Determine style
    const style = this.determineStyle(note);
    
    return {
      noteIndex: index,
      finger: note.fingering!,
      x,
      y,
      placement,
      hand: note.hand,
      style
    };
  }

  private determineVerticalPlacement(note: Note): 'above' | 'below' {
    // Rule 1: Based on staff (grand staff convention)
    if (note.staff === 1) {
      return 'above'; // Treble clef - RH - above
    } else if (note.staff === 2) {
      return 'below'; // Bass clef - LH - below
    }
    
    // Rule 2: Based on stem direction
    if (note.stem === 'up') {
      return 'above';
    } else if (note.stem === 'down') {
      return 'below';
    }
    
    // Rule 3: Based on hand
    return note.hand === 'RH' ? 'above' : 'below';
  }

  private calculateXPosition(note: Note, index: number): number {
    // Base position centered on note
    // In actual rendering, this would be calculated from SVG coordinates
    return index * 30 + 50;
  }

  private calculateYPosition(note: Note, placement: 'above' | 'below'): number {
    // Calculate Y based on pitch and placement
    // Middle C (60) is at y = 200 (example)
    const baseY = 200 - (note.pitch - 60) * 5;
    const offset = placement === 'above' ? -20 : 20;
    return baseY + offset;
  }

  private determineStyle(note: Note): FingeringPlacement['style'] {
    const finger = note.fingering!;
    
    return {
      fontSize: note.isGrace ? '9pt' : '11pt',
      fontWeight: note.hasAccent ? '600' : '400',
      color: this.colorMode ? this.fingerColors[finger] : '#000000'
    };
  }

  private resolveCollisions(placements: FingeringPlacement[]): FingeringPlacement[] {
    const minHorizontalGap = 15;
    const minVerticalGap = 12;
    
    for (let i = 1; i < placements.length; i++) {
      const prev = placements[i - 1];
      const curr = placements[i];
      
      const horizontalDist = Math.abs(curr.x - prev.x);
      const verticalDist = Math.abs(curr.y - prev.y);
      
      // Check for collision
      if (horizontalDist < minHorizontalGap && verticalDist < minVerticalGap) {
        // Offset current placement
        if (curr.placement === 'above') {
          curr.y -= minVerticalGap - verticalDist;
        } else {
          curr.y += minVerticalGap - verticalDist;
        }
      }
    }
    
    return placements;
  }

  // Generate SVG for fingering display
  renderFingeringSVG(placements: FingeringPlacement[], width: number, height: number): string {
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}">`;
    
    placements.forEach(p => {
      if (this.colorMode) {
        // Add background circle for educational mode
        svg += `
          <circle 
            cx="${p.x}" 
            cy="${p.y}" 
            r="8" 
            fill="${this.fingerColors[p.finger]}20"
            stroke="${p.style.color}"
            stroke-width="1"
          />`;
      }
      
      svg += `
        <text 
          x="${p.x}" 
          y="${p.y}" 
          text-anchor="middle" 
          dominant-baseline="central"
          font-family="Times New Roman, serif"
          font-size="${p.style.fontSize}"
          font-weight="${p.style.fontWeight}"
          fill="${p.style.color}"
        >${p.finger}</text>`;
    });
    
    svg += '</svg>';
    return svg;
  }

  // Get finger color for UI display
  getFingerColor(finger: Finger): string {
    return this.colorMode ? this.fingerColors[finger] : '#000000';
  }

  // Get finger name
  getFingerName(finger: Finger): string {
    const names: Record<Finger, string> = {
      1: 'Thumb',
      2: 'Index',
      3: 'Middle',
      4: 'Ring',
      5: 'Pinky'
    };
    return names[finger];
  }
}

export const fingeringRenderer = new FingeringRenderer();
