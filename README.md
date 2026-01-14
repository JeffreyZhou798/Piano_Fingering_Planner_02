# 🎹 Piano Fingering Planner

An AI-powered web application that automatically generates optimal piano fingering suggestions for MusicXML scores. Built with Vue 3, TypeScript, and a dual-layer architecture combining pattern recognition with dynamic programming optimization.

## ✨ Features

### Core Functionality
- **MusicXML Support**: Upload `.mxl`, `.musicxml`, or `.xml` piano scores
- **Automatic Fingering Generation**: AI-powered fingering suggestions for both hands
- **Pattern Recognition**: Identifies 11 musical pattern types (scales, arpeggios, chords, etc.)
- **Manual Editing**: Click any note to adjust the suggested fingering
- **Export**: Download annotated MusicXML with fingering included

### 🔗 Live Demo

  **Try it Now**: https://piano-fingering-planner-02.vercel.app/

### Musical Intelligence
- **11 Pattern Types Recognized**:
  - Primary: SCALE, ARPEGGIO, REPEATED, LEAP, CHORDAL, MELODIC, UNKNOWN
  - Special: ALBERTI, ORNAMENTED, OSTINATO, POLYPHONIC
- **Rule-Based Optimization**: Based on established piano pedagogy
- **Difficulty Levels**: Beginner, Intermediate, and Advanced modes
- **Validated Algorithm**: Tested against 27 real pieces with professional fingerings

### User Experience
- **Modern UI**: Clean, responsive design
- **Dark Mode**: Toggle between light and dark themes
- **Color Mode**: Color-coded fingers for educational purposes
- **Real-time Feedback**: Toast notifications for all operations

## 🏗️ Model Architecture

The system uses a **dual-layer architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    MusicXML Input                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Pattern Recognition (Decision Tree)               │
│  ─────────────────────────────────────────────────────────  │
│  • Analyzes musical features (pitch, rhythm, harmony)       │
│  • Classifies note sequences into 11 pattern types          │
│  • Uses sliding window with adaptive sizing                 │
│  • File: src/core/PatternRecognizer.ts                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Fingering Optimization (Rule-based DP)            │
│  ─────────────────────────────────────────────────────────  │
│  • Models fingering as shortest-path problem                │
│  • Four-level cost function:                                │
│    1. Physical Constraints (weight 100-150)                 │
│    2. Movement Efficiency (weight 50-70)                    │
│    3. Musical Constraints (weight 30-60)                    │
│    4. Pattern-Specific Rules (weight 20-50)                 │
│  • File: src/core/FingeringPlanner.ts                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Fingering Renderer                                │
│  ─────────────────────────────────────────────────────────  │
│  • Calculates visual placement for fingering numbers        │
│  • File: src/core/FingeringRenderer.ts                      │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm Details

#### Layer 1: Decision Tree Pattern Recognition
- **Input**: Sequence of notes with pitch, duration, articulation
- **Output**: Pattern type classification (SCALE, ARPEGGIO, etc.)
- **Method**: Feature extraction + rule-based decision tree
- **Complexity**: O(n) with sliding window

#### Layer 2: Dynamic Programming Optimization
- **Input**: Notes + Pattern context
- **Output**: Optimal finger assignment (1-5) for each note
- **Method**: Viterbi-style DP with 4-level cost function
- **Complexity**: O(n × f²) where n = notes, f = 5 fingers

### Validation Results

Tested against 27 professional fingered pieces (9,178 fingerings):

| Category | Accuracy | Examples |
|----------|----------|----------|
| Finger Exercises | 60-85% | Hanon-style exercises |
| Classical Pieces | 40-50% | Mozart, Bach, Chopin |
| Pop/Contemporary | 45-50% | Various arrangements |

**Overall Accuracy: ~50%** (improved from initial 33%)

Key findings from validation:
- Finger 4 usage: 16.4% (matches professional fingerings)
- Most common transitions: 2→1 (936), 1→2 (900), 4→5 (590), 5→4 (561)
- Five-finger position patterns work well for simple pieces
- Scale patterns require specialized thumb-crossing logic

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/piano-fingering-planner.git
cd piano-fingering-planner

# Install dependencies
npm install

# Start development server
npm run dev
```

### Build for Production

```bash
npm run build
```

## 📁 Project Structure

```
piano-fingering-planner/
├── src/
│   ├── components/          # Vue components
│   │   ├── Header.vue
│   │   ├── UploadSection.vue
│   │   ├── ScoreInfo.vue
│   │   ├── ControlPanel.vue
│   │   ├── ScoreViewer.vue
│   │   ├── NoteCard.vue
│   │   └── ToastContainer.vue
│   ├── core/                # Core algorithms
│   │   ├── MusicXMLParser.ts    # XML parsing & writing
│   │   ├── PatternRecognizer.ts # Layer 1: Decision Tree
│   │   ├── FingeringPlanner.ts  # Layer 2: Rule-based DP
│   │   └── FingeringRenderer.ts # Layer 3: Visual placement
│   ├── stores/              # Pinia state management
│   │   └── appStore.ts
│   ├── styles/              # Global CSS
│   │   └── main.css
│   ├── types/               # TypeScript definitions
│   │   └── music.ts
│   ├── App.vue
│   └── main.ts
├── public/
│   └── piano.svg
├── scripts/                 # Validation scripts
│   └── validate-fingerings.ts
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── vercel.json
```

## 🎯 Usage

1. **Upload**: Click or drag-and-drop a MusicXML file (.mxl, .musicxml, .xml)
2. **Configure**: Select difficulty level (Beginner/Intermediate/Advanced)
3. **Generate**: Click "Generate Fingering" to analyze and plan
4. **Review**: View results by measure or as a list
5. **Edit**: Click any fingering number to change it manually
6. **Download**: Export the annotated MusicXML file

## 🚢 Deployment

### Vercel (Recommended)
1. Push to GitHub
2. Import project in Vercel
3. Deploy automatically (uses `vercel.json` configuration)

### Manual Deployment
```bash
npm run build
# Upload dist/ folder to any static hosting
```

## 📦 Files for GitHub

### Required Files (Must Upload)
```
├── src/                     # All source code
├── public/                  # Static assets
├── index.html               # Entry HTML
├── package.json             # Dependencies
├── package-lock.json        # Lock file
├── tsconfig.json            # TypeScript config
├── tsconfig.node.json       # Node TypeScript config
├── vite.config.ts           # Vite config
├── vercel.json              # Vercel deployment config
├── .gitignore               # Git ignore rules
└── README.md                # Documentation
```

### Optional Files (Can Skip)
```
├── node_modules/            # Auto-installed (in .gitignore)
├── dist/                    # Build output (in .gitignore)
├── scripts/                 # Development/validation scripts
├── 钢琴乐曲库带指法/         # Test data (large)
├── CompositionExamples/     # Test data (large)
├── 项目蓝图/                 # Design documents (Chinese)
├── *.tsbuildinfo            # TypeScript cache
└── vite.config.js           # Generated JS config
```

## 🔧 Technical Details

### Technologies
- **Frontend**: Vue 3 (Composition API)
- **Language**: TypeScript
- **State Management**: Pinia
- **Build Tool**: Vite
- **XML Processing**: JSZip (for .mxl files)

### Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 📚 Educational Value

This tool is designed to:
- **Reduce Cognitive Load**: Let learners focus on music, not finger decisions
- **Teach Good Habits**: Based on established piano pedagogy
- **Explain Decisions**: Every fingering choice has a musical reason
- **Support Teachers**: Generate examples quickly, then customize

## 🙏 Credits

- Piano pedagogy rules based on classical teaching methods (Hanon, Czerny, Schmitt)
- MusicXML parsing follows the MusicXML 4.0 specification
- Algorithm validated against professional fingered editions

---

**Note**: This is an educational tool. While the AI provides intelligent suggestions, professional pianists and teachers should review fingerings for performance-critical applications.


## ⚠️ Copyright Notice

© 2026 Jeffrey Zhou. All rights reserved.

This repository and its contents are protected by copyright law.  
No part of this project may be copied, reproduced, modified, or distributed without prior written permission from the author.

Commercial use is strictly prohibited.

*Built with ❤️ for music education*


