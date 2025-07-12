
/**
 * @fileoverview Control real time music with text prompts
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {css, CSSResultGroup, html, LitElement, svg, TemplateResult, unsafeCSS} from 'lit';
import {customElement, property, query, state} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';
import {styleMap}from 'lit/directives/style-map.js';

import {
  GoogleGenAI,
  type GenerateContentResponse, 
  type LiveMusicGenerationConfig,
  type LiveMusicServerMessage,
  type LiveMusicSession,
  type WeightedPrompt,
} from '@google/genai';
import {decode, decodeAudioData} from './utils';

// Lyria AI instance
const lyriaAI = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  apiVersion: 'v1alpha',
});
const lyriaModel = 'lyria-realtime-exp';

// Gemini Text Generation AI instance (can be used by simulated agents if needed)
// const textGenAI = new GoogleGenAI({apiKey: process.env.API_KEY}); // Example if needed
// const GEMINI_TEXT_MODEL = 'gemini-2.5-flash-preview-04-17';


// -----------------------------------------------------------------------------
// SECTION: Model Context Protocol (MCP) Interfaces
// -----------------------------------------------------------------------------
interface MCPPerformanceMetrics {
  total_cycles: number;
  successful_cycles: number;
  avg_cycle_time: number; // in ms
  efficiency_score: number; // 0.0 to 1.0
}

interface MCPInstructionsConstraints {
  hr_target_zone: [number, number];
  max_bpm_change_per_cycle: number;
  fatigue_threshold: number; // 0.0 to 1.0
  mood_adaptation_sensitivity: number; // 0.0 to 1.0
}

interface MCPInstructions {
  primary_goal: string;
  adaptive_strategy: string;
  style_preferences: string[];
  constraints: MCPInstructionsConstraints;
}

interface MCPAgentPerformance {
  success_rate: number;
  avg_response_time: number; // in ms
}

interface MCPAgent {
  id: string;
  status: 'ready' | 'working' | 'error' | 'success';
  specialization: string;
  performance: MCPAgentPerformance;
  last_execution: string | null; // ISO timestamp
  reasoning?: string;
  output_summary?: string | Record<string, any>;
}

interface MCPRawData {
  hr: number;
  cadence: number;
  motion: number; // e.g., accelerometer data
}

interface MCPProcessedMetrics {
  heart_rate_bpm: number;
  hr_zone: number; // 1-5
  cadence_spm: number;
  user_fatigue_score: number; // 0.0 to 1.0
  session_minute: number;
  current_mood: string;
  trend_direction: 'increasing' | 'decreasing' | 'stable';
}

interface MCPContextAnalysis {
  workout_phase: 'warmup' | 'main' | 'cooldown' | 'idle';
  energy_level: 'low' | 'moderate' | 'high';
  adaptation_needed: boolean;
  risk_factors: string[];
  opportunities: string[];
  recommendations?: string[];
}

interface MCPSystemState {
  raw_data: MCPRawData;
  processed_metrics: MCPProcessedMetrics;
  context_analysis: MCPContextAnalysis;
}

interface MCPUserProfile {
  preferred_bpm_range: [number, number];
  instrument_preferences: Record<string, number>; // e.g., { bass: 0.9, drums: 0.8 }
  style_affinity: Record<string, number>; // e.g., { "Trip Hop": 0.8 }
}

interface MCPSessionHistoryItem {
  timestamp: string;
  metrics: MCPProcessedMetrics;
  prompt_sent_to_lyria: MCPPromptAndConfig;
}

interface MCPLearnedPatterns {
  hr_response_to_bpm: Record<string, any>;
  fatigue_indicators: string[];
  effective_transitions: string[];
}

interface MCPMemory {
  user_profile: MCPUserProfile;
  session_history: MCPSessionHistoryItem[];
  learned_patterns: MCPLearnedPatterns;
}

interface MCPPromptAndConfig {
  weightedPrompts: WeightedPrompt[]; 
  config: Partial<LiveMusicGenerationConfig>; 
}


interface MCPLyriaOutput {
  current_prompt: MCPPromptAndConfig;
  generation_history: MCPPromptAndConfig[]; 
  effectiveness_scores: number[]; 
}

interface MCPFeedbackLoop {
  pattern_detection: {
    identified_patterns: string[];
    confidence_scores: number[];
  };
  adaptation_suggestions: string[];
  learning_insights: string[];
}

interface ModelContextProtocol {
  context_schema_version: string;
  session_id: string;
  timestamp: string; // ISO timestamp of last MCP update
  performance_metrics: MCPPerformanceMetrics;
  instructions: MCPInstructions;
  agents: {
    ingestion: MCPAgent;
    context: MCPAgent;
    musical: MCPAgent;
    feedback: MCPAgent;
  };
  system_state: MCPSystemState;
  memory: MCPMemory;
  lyria_output: MCPLyriaOutput;
  feedback_loop: MCPFeedbackLoop;
}
// --- End MCP Interfaces ---

// Agent API Payloads (for simulated API calls)
interface IngestionAgentInput { raw_data: MCPRawData; current_session_minute: number; }
interface IngestionAgentOutput { processed_metrics: MCPProcessedMetrics; reasoning: string; output_summary: Partial<MCPProcessedMetrics>; }

interface ContextAgentInput { processed_metrics: MCPProcessedMetrics; fatigue_threshold: number; simulation_running: boolean; }
interface ContextAgentOutput { context_analysis: MCPContextAnalysis; reasoning: string; output_summary: Partial<MCPContextAnalysis>; }

interface MusicalAgentInput {
    context_analysis: MCPContextAnalysis;
    processed_metrics: MCPProcessedMetrics;
    style_preferences: string[];
    total_cycles: number; // for varying style
}
interface MusicalAgentOutput { lyria_prompt: MCPPromptAndConfig; reasoning: string; output_summary: {prompt_summary: string; config_summary: string}; }

interface FeedbackAgentInput { /* Define as needed */ }
interface FeedbackAgentOutput { reasoning: string; output_summary: string; updated_memory?: Partial<MCPMemory> }


interface Prompt { 
  readonly promptId: string;
  readonly color: string;
  text: string;
  weight: number;
  isTextEditable?: boolean;
}

type PlaybackState = 'stopped' | 'playing' | 'loading' | 'paused';

const AGENT_KEYS = {
    INGESTION: 'ingestion',
    CONTEXT: 'context',
    MUSICAL: 'musical',
    FEEDBACK: 'feedback',
} as const;
type AgentKey = typeof AGENT_KEYS[keyof typeof AGENT_KEYS];


// -----------------------------------------------------------------------------
// SECTION: Utility Functions and Constants
// -----------------------------------------------------------------------------

function throttle(func: (...args: unknown[]) => void, delay: number) {
  let lastCall = 0;
  return (...args: unknown[]) => {
    const now = Date.now();
    const timeSinceLastCall = now - lastCall;
    if (timeSinceLastCall >= delay) {
      func(...args);
      lastCall = now;
    }
  };
}

const PROMPT_TEXT_PRESETS = [
  'Bossa Nova', 'Minimal Techno', 'Drum and Bass', 'Post Punk', 'Shoegaze',
  'Funk', 'Chiptune', 'Lush Strings', 'Sparkling Arpeggios', 'Staccato Rhythms',
  'Punchy Kick', 'Dubstep', 'K Pop', 'Neo Soul', 'Trip Hop', 'Thrash',
];

const COLORS = [
  '#9900ff', '#5200ff', '#ff25f6', '#2af6de', '#ffdd28',
  '#3dffab', '#d8ff3e', '#d9b2ff',
];

function getUnusedRandomColor(usedColors: string[]): string {
  const availableColors = COLORS.filter((c) => !usedColors.includes(c));
  if (availableColors.length === 0) {
    return COLORS[Math.floor(Math.random() * COLORS.length)];
  }
  return availableColors[Math.floor(Math.random() * availableColors.length)];
}

// -----------------------------------------------------------------------------
// SECTION: Small UI Components / Building Blocks
// -----------------------------------------------------------------------------

@customElement('weight-slider')
class WeightSlider extends LitElement {
  static override styles = css`
    :host {
      cursor: ns-resize;
      position: relative;
      height: 100%;
      display: flex;
      justify-content: center;
      flex-direction: column;
      align-items: center;
      padding: 5px;
    }
    .scroll-container {
      width: 100%;
      flex-grow: 1;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
    .value-display {
      font-size: 1.3vmin;
      color: #ccc;
      margin: 0.5vmin 0;
      user-select: none;
      text-align: center;
    }
    .slider-container {
      position: relative;
      width: 10px;
      height: 100%;
      background-color: #0009;
      border-radius: 4px;
    }
    #thumb {
      position: absolute;
      bottom: 0;
      left: 0;
      width: 100%;
      border-radius: 4px;
      box-shadow: 0 0 3px rgba(0, 0, 0, 0.7);
    }
  `;

  @property({type: Number}) value = 0; // Range 0-2
  @property({type: String}) color = '#000';

  @query('.scroll-container') private scrollContainer!: HTMLDivElement;

  private dragStartPos = 0;
  private dragStartValue = 0;
  private containerBounds: DOMRect | null = null;

  constructor() {
    super();
    this.handlePointerDown = this.handlePointerDown.bind(this);
    this.handlePointerMove = this.handlePointerMove.bind(this);
    this.handleTouchMove = this.handleTouchMove.bind(this);
    this.handlePointerUp = this.handlePointerUp.bind(this);
  }

  private handlePointerDown(e: PointerEvent) {
    e.preventDefault();
    this.containerBounds = this.scrollContainer.getBoundingClientRect();
    this.dragStartPos = e.clientY;
    this.dragStartValue = this.value;
    document.body.classList.add('dragging');
    window.addEventListener('pointermove', this.handlePointerMove);
    window.addEventListener('touchmove', this.handleTouchMove, {
      passive: false,
    });
    window.addEventListener('pointerup', this.handlePointerUp, {once: true});
    this.updateValueFromPosition(e.clientY);
  }

  private handlePointerMove(e: PointerEvent) {
    this.updateValueFromPosition(e.clientY);
  }

  private handleTouchMove(e: TouchEvent) {
    e.preventDefault();
    this.updateValueFromPosition(e.touches[0].clientY);
  }

  private handlePointerUp(e: PointerEvent) {
    window.removeEventListener('pointermove', this.handlePointerMove);
    document.body.classList.remove('dragging');
    this.containerBounds = null;
  }

  private handleWheel(e: WheelEvent) {
    e.preventDefault();
    const delta = e.deltaY;
    this.value = this.value + delta * -0.005;
    this.value = Math.max(0, Math.min(2, this.value));
    this.dispatchInputEvent();
  }

  private updateValueFromPosition(clientY: number) {
    if (!this.containerBounds) return;

    const trackHeight = this.containerBounds.height;
    const relativeY = clientY - this.containerBounds.top;
    const normalizedValue =
      1 - Math.max(0, Math.min(trackHeight, relativeY)) / trackHeight;
    this.value = normalizedValue * 2;

    this.dispatchInputEvent();
  }

  private dispatchInputEvent() {
    this.dispatchEvent(new CustomEvent<number>('input', {detail: this.value}));
  }

  override render() {
    const thumbHeightPercent = (this.value / 2) * 100;
    const thumbStyle = styleMap({
      height: `${thumbHeightPercent}%`,
      backgroundColor: this.color,
      display: this.value > 0.01 ? 'block' : 'none',
    });
    const displayValue = this.value.toFixed(2);

    return html`
      <div
        class="scroll-container"
        @pointerdown=${this.handlePointerDown}
        @wheel=${this.handleWheel}>
        <div class="slider-container">
          <div id="thumb" style=${thumbStyle}></div>
        </div>
        <div class="value-display">${displayValue}</div>
      </div>
    `;
  }
}

class IconButton extends LitElement {
  static override styles = css`
    :host {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: none;
    }
    :host(:hover) svg {
      transform: scale(1.2);
    }
    svg {
      width: 100%;
      height: 100%;
      transition: transform 0.5s cubic-bezier(0.25, 1.56, 0.32, 0.99);
    }
    .hitbox {
      pointer-events: all;
      position: absolute;
      width: 65%;
      aspect-ratio: 1;
      top: 9%;
      border-radius: 50%;
      cursor: pointer;
    }
  ` as CSSResultGroup;

  protected renderIcon() {
    return svg``;
  }

  private renderSVG() {
    return html` <svg
      width="140"
      height="140"
      viewBox="0 -10 140 150"
      fill="none"
      xmlns="http://www.w3.org/2000/svg">
      <rect
        x="22"
        y="6"
        width="96"
        height="96"
        rx="48"
        fill="black"
        fill-opacity="0.05" />
      <rect
        x="23.5"
        y="7.5"
        width="93"
        height="93"
        rx="46.5"
        stroke="black"
        stroke-opacity="0.3"
        stroke-width="3" />
      <g filter="url(#filter0_ddi_1048_7373)">
        <rect
          x="25"
          y="9"
          width="90"
          height="90"
          rx="45"
          fill="white"
          fill-opacity="0.05"
          shape-rendering="crispEdges" />
      </g>
      ${this.renderIcon()}
      <defs>
        <filter
          id="filter0_ddi_1048_7373"
          x="0"
          y="0"
          width="140"
          height="140"
          filterUnits="userSpaceOnUse"
          color-interpolation-filters="sRGB">
          <feFlood flood-opacity="0" result="BackgroundImageFix" />
          <feColorMatrix
            in="SourceAlpha"
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
            result="hardAlpha" />
          <feOffset dy="2" />
          <feGaussianBlur stdDeviation="4" />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
          <feBlend
            mode="normal"
            in2="BackgroundImageFix"
            result="effect1_dropShadow_1048_7373" />
          <feColorMatrix
            in="SourceAlpha"
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
            result="hardAlpha" />
          <feOffset dy="16" />
          <feGaussianBlur stdDeviation="12.5" />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
          <feBlend
            mode="normal"
            in2="effect1_dropShadow_1048_7373"
            result="effect2_dropShadow_1048_7373" />
          <feBlend
            mode="normal"
            in="SourceGraphic"
            in2="effect2_dropShadow_1048_7373"
            result="shape" />
          <feColorMatrix
            in="SourceAlpha"
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
            result="hardAlpha" />
          <feOffset dy="3" />
          <feGaussianBlur stdDeviation="1.5" />
          <feComposite in2="hardAlpha" operator="arithmetic" k2="-1" k3="1" />
          <feColorMatrix
            type="matrix"
            values="0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.05 0" />
          <feBlend
            mode="normal"
            in2="shape"
            result="effect3_innerShadow_1048_7373" />
        </filter>
      </defs>
    </svg>`;
  }

  override render() {
    return html`${this.renderSVG()}<div class="hitbox"></div>`;
  }
}

@customElement('play-pause-button')
export class PlayPauseButton extends IconButton {
  @property({type: String}) playbackState: PlaybackState = 'stopped';

  static override styles = [
    IconButton.styles,
    css`
      .loader {
        stroke: #ffffff;
        stroke-width: 3;
        stroke-linecap: round;
        animation: spin linear 1s infinite;
        transform-origin: center;
        transform-box: fill-box;
      }
      @keyframes spin {
        from {
          transform: rotate(0deg);
        }
        to {
          transform: rotate(359deg);
        }
      }
    `,
  ];

  private renderPause() {
    return svg`<path
      d="M75.0037 69V39H83.7537V69H75.0037ZM56.2537 69V39H65.0037V69H56.2537Z"
      fill="#FEFEFE"
    />`;
  }

  private renderPlay() {
    return svg`<path d="M60 71.5V36.5L87.5 54L60 71.5Z" fill="#FEFEFE" />`;
  }

  private renderLoading() {
    return svg`<path shape-rendering="crispEdges" class="loader" d="M70,74.2L70,74.2c-10.7,0-19.5-8.7-19.5-19.5l0,0c0-10.7,8.7-19.5,19.5-19.5
            l0,0c10.7,0,19.5,8.7,19.5,19.5l0,0"/>`;
  }

  override renderIcon() {
    if (this.playbackState === 'playing') {
      return this.renderPause();
    } else if (this.playbackState === 'loading') {
      return this.renderLoading();
    } else {
      return this.renderPlay();
    }
  }
}

@customElement('reset-button')
export class ResetButton extends IconButton {
  private renderResetIcon() {
    return svg`<path fill="#fefefe" d="M71,77.1c-2.9,0-5.7-0.6-8.3-1.7s-4.8-2.6-6.7-4.5c-1.9-1.9-3.4-4.1-4.5-6.7c-1.1-2.6-1.7-5.3-1.7-8.3h4.7
      c0,4.6,1.6,8.5,4.8,11.7s7.1,4.8,11.7,4.8c4.6,0,8.5-1.6,11.7-4.8c3.2-3.2,4.8-7.1,4.8-11.7s-1.6-8.5-4.8-11.7
      c-3.2-3.2-7.1-4.8-11.7-4.8h-0.4l3.7,3.7L71,46.4L61.5,37l9.4-9.4l3.3,3.4l-3.7,3.7H71c2.9,0,5.7,0.6,8.3,1.7
      c2.6,1.1,4.8,2.6,6.7,4.5c1.9,1.9,3.4,4.1,4.5,6.7c1.1,2.6,1.7,5.3,1.7,8.3c0,2.9-0.6,5.7-1.7,8.3c-1.1,2.6-2.6,4.8-4.5,6.7
      s-4.1,3.4-6.7,4.5C76.7,76.5,73.9,77.1,71,77.1z"/>`;
  }

  override renderIcon() {
    return this.renderResetIcon();
  }
}

@customElement('add-prompt-button')
export class AddPromptButton extends IconButton {
  private renderAddIcon() {
    return svg`<path d="M67 40 H73 V52 H85 V58 H73 V70 H67 V58 H55 V52 H67 Z" fill="#FEFEFE" />`;
  }

  override renderIcon() {
    return this.renderAddIcon();
  }
}

@customElement('toast-message')
class ToastMessage extends LitElement {
  static override styles = css`
    .toast {
      line-height: 1.6;
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      background-color: #000;
      color: white;
      padding: 15px;
      border-radius: 5px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 15px;
      min-width: 200px;
      max-width: 80vw;
      transition: transform 0.5s cubic-bezier(0.19, 1, 0.22, 1);
      z-index: 11;
    }
    button {
      border-radius: 100px;
      aspect-ratio: 1;
      border: none;
      color: #000;
      cursor: pointer;
    }
    .toast:not(.showing) {
      transition-duration: 1s;
      transform: translate(-50%, -200%);
    }
  `;

  @property({type: String}) message = '';
  @property({type: Boolean}) showing = false;

  override render() {
    return html`<div class=${classMap({showing: this.showing, toast: true})}>
      <div class="message">${this.message}</div>
      <button @click=${this.hide}>✕</button>
    </div>`;
  }

  show(message: string) {
    this.showing = true;
    this.message = message;
  }

  hide() {
    this.showing = false;
  }
}

@customElement('mcp-agent-card')
class MCPAgentCard extends LitElement {
    static override styles = css`
        :host {
            display: block;
            background-color: #333;
            border-radius: 8px;
            padding: 1em;
            border-left: 5px solid var(--agent-color, #888);
            font-size: 0.9em;
            margin-bottom: 1em;
        }
        h4 { margin-top: 0; margin-bottom: 0.5em; color: var(--agent-color, #eee); }
        .status {
            font-weight: bold;
            text-transform: capitalize;
            margin-bottom: 0.5em;
        }
        .status-ready { color: #8f8; } /* Light green */
        .status-working { color: #ff8; } /* Yellow */
        .status-success { color: #8f8; } /* Light green */
        .status-error { color: #f88; } /* Light red */
        pre {
            background-color: #222;
            padding: 0.5em;
            border-radius: 4px;
            max-height: 100px;
            overflow-y: auto;
            font-size: 0.9em;
            word-break: break-all;
            white-space: pre-wrap;
        }
        .label { color: #aaa; }
    `;

    @property({type: Object}) agent!: MCPAgent;
    @property({type: String}) agentName = '';
    @property({type: String}) agentColor = '#888';

    override render() {
        if (!this.agent) return html`Loading agent...`;
        const statusClass = `status-${this.agent.status}`;
        let displayTime = 'N/A';
        if (this.agent.last_execution) {
            try {
                displayTime = new Date(this.agent.last_execution).toLocaleTimeString();
            } catch (e) {
                console.warn("Invalid date for last_execution:", this.agent.last_execution);
                displayTime = "Invalid Date";
            }
        }

        return html`
            <style>:host { --agent-color: ${unsafeCSS(this.agentColor)}; }</style>
            <h4>${this.agentName.toUpperCase()} Agent <span class="status ${statusClass}">(${this.agent.status})</span></h4>
            <p><span class="label">Specialization:</span> ${this.agent.specialization}</p>
            ${this.agent.output_summary ? html`<div><span class="label">Output:</span> <pre>${typeof this.agent.output_summary === 'string' ? this.agent.output_summary : JSON.stringify(this.agent.output_summary, null, 2)}</pre></div>` : ''}
            ${this.agent.reasoning ? html`<div><span class="label">Reasoning:</span> <pre>${this.agent.reasoning}</pre></div>` : ''}
            <p><span class="label">Last Run:</span> ${displayTime}</p>
        `;
    }
}

// -----------------------------------------------------------------------------
// SECTION: More Complex UI Controllers / Components
// -----------------------------------------------------------------------------

@customElement('prompt-controller')
class PromptController extends LitElement {
  static override styles = css`
    .prompt {
      position: relative;
      height: 100%;
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      box-sizing: border-box;
      overflow: hidden;
      background-color: #2a2a2a;
      border-radius: 5px;
    }
    .remove-button {
      position: absolute;
      top: 1.2vmin;
      left: 1.2vmin;
      background: #666;
      color: #fff;
      border: none;
      border-radius: 50%;
      width: 2.8vmin;
      height: 2.8vmin;
      font-size: 1.8vmin;
      display: flex;
      align-items: center;
      justify-content: center;
      line-height: 2.8vmin;
      cursor: pointer;
      opacity: 0.5;
      transition: opacity 0.2s;
      z-index: 10;
    }
    .remove-button:hover {
      opacity: 1;
    }
    weight-slider {
      max-height: calc(100% - 9vmin);
      flex: 1;
      min-height: 10vmin;
      width: 100%;
      box-sizing: border-box;
      overflow: hidden;
      margin: 2vmin 0 1vmin;
    }
    .controls {
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      align-items: center;
      gap: 0.2vmin;
      width: 100%;
      height: 8vmin;
      padding: 0 0.5vmin;
      box-sizing: border-box;
      margin-bottom: 1vmin;
    }
    #text {
      font-family: 'Google Sans', sans-serif;
      font-size: 1.8vmin;
      width: 100%;
      flex-grow: 1;
      max-height: 100%;
      padding: 0.4vmin;
      box-sizing: border-box;
      text-align: center;
      word-wrap: break-word;
      overflow-y: auto;
      border: none;
      outline: none;
      -webkit-font-smoothing: antialiased;
      color: #fff;
      scrollbar-width: thin;
      scrollbar-color: #666 #1a1a1a;
      background-color: #1e1e1e;
    }
    #text[contenteditable="false"] {
        background-color: #2a2a2a; 
        cursor: default;
    }
    #text::-webkit-scrollbar {
      width: 6px;
    }
    #text::-webkit-scrollbar-track {
      background: #0009;
      border-radius: 3px;
    }
    #text::-webkit-scrollbar-thumb {
      background-color: #666;
      border-radius: 3px;
    }
    :host([filtered='true']) #text {
      background: #da2000;
    }
  `;

  @property({type: String, reflect: true}) promptId = '';
  @property({type: String}) text = '';
  @property({type: Number}) weight = 0;
  @property({type: String}) color = '';
  @property({type: Boolean}) isTextEditable = true;


  @query('weight-slider') private weightInput!: WeightSlider;
  @query('#text') private textInput!: HTMLSpanElement;

  private handleTextKeyDown(e: KeyboardEvent) {
    if (!this.isTextEditable) {
        e.preventDefault();
        return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      this.updateText();
      (e.target as HTMLElement).blur();
    }
  }

  private dispatchPromptChange() {
    this.dispatchEvent(
      new CustomEvent<Prompt>('prompt-changed', {
        detail: {
          promptId: this.promptId,
          text: this.text,
          weight: this.weight,
          color: this.color,
          isTextEditable: this.isTextEditable,
        },
      }),
    );
  }

  private updateText() {
    if (!this.isTextEditable) return;
    const newText = this.textInput.textContent?.trim();
    if (newText === '') {
      this.textInput.textContent = this.text;
      return;
    }
    this.text = newText ?? this.text;
    this.dispatchPromptChange();
  }

  private updateWeight() {
    this.weight = this.weightInput.value;
    this.dispatchPromptChange();
  }

  private dispatchPromptRemoved() {
    this.dispatchEvent(
      new CustomEvent<string>('prompt-removed', {
        detail: this.promptId,
        bubbles: true,
        composed: true,
      }),
    );
  }

  override render() {
    const classes = classMap({
      'prompt': true,
    });
    return html`<div class=${classes}>
      <button class="remove-button" @click=${this.dispatchPromptRemoved}
        >×</button
      >
      <weight-slider
        id="weight"
        value=${this.weight}
        color=${this.color}
        @input=${this.updateWeight}></weight-slider>
      <div class="controls">
        <span
          id="text"
          spellcheck="false"
          contenteditable=${this.isTextEditable ? 'plaintext-only' : 'false'}
          @keydown=${this.handleTextKeyDown}
          @blur=${this.updateText}
          >${this.text}</span
        >
      </div>
    </div>`;
  }
}

@customElement('settings-controller')
class SettingsController extends LitElement {
  static override styles = css`
    :host {
      display: block;
      padding: 2vmin;
      background-color: #2a2a2a;
      color: #eee;
      box-sizing: border-box;
      border-radius: 5px;
      font-family: 'Google Sans', sans-serif;
      font-size: 1.5vmin;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: #666 #1a1a1a;
      transition: width 0.3s ease-out max-height 0.3s ease-out;
    }
    :host([showadvanced]) {
      max-height: 40vmin;
    }
    :host::-webkit-scrollbar {
      width: 6px;
    }
    :host::-webkit-scrollbar-track {
      background: #1a1a1a;
      border-radius: 3px;
    }
    :host::-webkit-scrollbar-thumb {
      background-color: #666;
      border-radius: 3px;
    }
    .setting {
      margin-bottom: 0.5vmin;
      display: flex;
      flex-direction: column;
      gap: 0.5vmin;
    }
    label {
      font-weight: bold;
      display: flex;
      justify-content: space-between;
      align-items: center;
      white-space: nowrap;
      user-select: none;
    }
    label span:last-child {
      font-weight: normal;
      color: #ccc;
      min-width: 3em;
      text-align: right;
    }
    input[type='range'] {
      --track-height: 8px;
      --track-bg: #0009;
      --track-border-radius: 4px;
      --thumb-size: 16px;
      --thumb-bg: #5200ff;
      --thumb-border-radius: 50%;
      --thumb-box-shadow: 0 0 3px rgba(0, 0, 0, 0.7);
      --value-percent: 0%;
      -webkit-appearance: none;
      appearance: none;
      width: 100%;
      height: var(--track-height);
      background: transparent;
      cursor: pointer;
      margin: 0.5vmin 0;
      border: none;
      padding: 0;
      vertical-align: middle;
    }
    input[type='range']::-webkit-slider-runnable-track {
      width: 100%;
      height: var(--track-height);
      cursor: pointer;
      border: none;
      background: linear-gradient(
        to right,
        var(--thumb-bg) var(--value-percent),
        var(--track-bg) var(--value-percent)
      );
      border-radius: var(--track-border-radius);
    }
    input[type='range']::-moz-range-track {
      width: 100%;
      height: var(--track-height);
      cursor: pointer;
      background: var(--track-bg);
      border-radius: var(--track-border-radius);
      border: none;
    }
    input[type='range']::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      height: var(--thumb-size);
      width: var(--thumb-size);
      background: var(--thumb-bg);
      border-radius: var(--thumb-border-radius);
      box-shadow: var(--thumb-box-shadow);
      cursor: pointer;
      margin-top: calc((var(--thumb-size) - var(--track-height)) / -2);
    }
    input[type='range']::-moz-range-thumb {
      height: var(--thumb-size);
      width: var(--thumb-size);
      background: var(--thumb-bg);
      border-radius: var(--thumb-border-radius);
      box-shadow: var(--thumb-box-shadow);
      cursor: pointer;
      border: none;
    }
    input[type='number'],
    input[type='text'],
    select,
    textarea { 
      background-color: #2a2a2a;
      color: #eee;
      border: 1px solid #666;
      border-radius: 3px;
      padding: 0.4vmin;
      font-size: 1.5vmin;
      font-family: inherit;
      box-sizing: border-box;
    }
    input[type='number'] {
      width: 6em;
    }
    input[type='text'],
    textarea { 
      width: 100%;
    }
    textarea { 
      min-height: 5vmin;
      resize: vertical;
      scrollbar-width: thin;
      scrollbar-color: #666 #1a1a1a;
    }
    textarea::-webkit-scrollbar {
      width: 6px;
    }
    textarea::-webkit-scrollbar-track {
      background: #1a1a1a;
      border-radius: 3px;
    }
    textarea::-webkit-scrollbar-thumb {
      background-color: #666;
      border-radius: 3px;
    }
    input[type='text']::placeholder,
    textarea::placeholder { 
      color: #888;
    }
    input[type='number']:focus,
    input[type='text']:focus,
    select:focus,
    textarea:focus { 
      outline: none;
      border-color: #5200ff;
      box-shadow: 0 0 0 2px rgba(82, 0, 255, 0.3);
    }
    select {
      width: 100%;
    }
    select:focus {
      outline: none;
      border-color: #5200ff;
    }
    select option {
      background-color: #2a2a2a;
      color: #eee;
    }
    .checkbox-setting {
      flex-direction: row;
      align-items: center;
      gap: 1vmin;
    }
    input[type='checkbox'] {
      cursor: pointer;
      accent-color: #5200ff;
    }
    .core-settings-row {
      display: flex;
      flex-direction: row;
      flex-wrap: wrap;
      gap: 4vmin;
      margin-bottom: 1vmin;
      justify-content: space-evenly;
    }
    .core-settings-row .setting {
      min-width: 16vmin;
    }
    .core-settings-row label span:last-child {
      min-width: 2.5em;
    }
    .advanced-toggle {
      cursor: pointer;
      margin: 2vmin 0 1vmin 0;
      color: #aaa;
      text-decoration: underline;
      user-select: none;
      font-size: 1.4vmin;
      width: fit-content;
    }
    .advanced-toggle:hover {
      color: #eee;
    }
    .advanced-settings {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(10vmin, 1fr));
      gap: 3vmin;
      overflow: hidden;
      max-height: 0;
      opacity: 0;
      transition:
        max-height 0.3s ease-out,
        opacity 0.3s ease-out;
    }
    .advanced-settings.visible {
      max-width: 120vmin;
      max-height: 40vmin;
      opacity: 1;
    }
    hr.divider {
      display: none;
      border: none;
      border-top: 1px solid #666;
      margin: 2vmin 0;
      width: 100%;
    }
    :host([showadvanced]) hr.divider {
      display: block;
    }
    .auto-row {
      display: flex;
      align-items: center;
      gap: 0.5vmin;
    }
    .setting[auto='true'] input[type='range'] {
      pointer-events: none;
      filter: grayscale(100%);
    }
    .auto-row span {
      margin-left: auto;
    }
    .auto-row label {
      cursor: pointer;
    }
    .auto-row input[type='checkbox'] {
      cursor: pointer;
      margin: 0;
    }
  `;

  private readonly defaultConfig: LiveMusicGenerationConfig = {
    temperature: 1.1,
    topK: 40,
    guidance: 4.0,
  };

  @state() private config: LiveMusicGenerationConfig = this.defaultConfig;

  @state() showAdvanced = false;
  @state() autoDensity = true;
  @state() lastDefinedDensity: number | undefined = undefined;
  @state() autoBrightness = true;
  @state() lastDefinedBrightness: number | undefined = undefined;


  public resetToDefaults() {
    this.config = {...this.defaultConfig};
    this.autoDensity = true;
    this.lastDefinedDensity = undefined;
    this.autoBrightness = true;
    this.lastDefinedBrightness = undefined;
    this.dispatchSettingsChange();
  }

  public setConfig(newConfig: Partial<LiveMusicGenerationConfig>) {
    const updatedConfig = {...this.config, ...newConfig};
    
    if (newConfig.density !== undefined) {
        this.autoDensity = false;
        this.lastDefinedDensity = newConfig.density;
    } else if (newConfig.density === undefined && this.autoDensity) {
        // Keep auto if newConfig explicitly sets density to undefined while auto is on
    }

    if (newConfig.brightness !== undefined) {
        this.autoBrightness = false;
        this.lastDefinedBrightness = newConfig.brightness;
    } else if (newConfig.brightness === undefined && this.autoBrightness) {
        // Keep auto
    }
    
    this.config = updatedConfig;
    this.requestUpdate(); 
  }


  override connectedCallback() {
    super.connectedCallback();
    if (this.config.density !== undefined) {
      this.lastDefinedDensity = this.config.density;
      this.autoDensity = false;
    }
    if (this.config.brightness !== undefined) {
      this.lastDefinedBrightness = this.config.brightness;
      this.autoBrightness = false;
    }
  }


  private updateSliderBackground(inputEl: HTMLInputElement) {
    if (inputEl.type !== 'range') {
      return;
    }
    const min = Number(inputEl.min) || 0;
    const max = Number(inputEl.max) || 100;
    const value = Number(inputEl.value);
    const percentage = ((value - min) / (max - min)) * 100;
    inputEl.style.setProperty('--value-percent', `${percentage}%`);
  }

  private handleInputChange(e: Event) {
    const target = e.target as HTMLInputElement;
    const key = target.id as
      | keyof LiveMusicGenerationConfig
      | 'auto-density'
      | 'auto-brightness';
    let value: string | number | boolean | undefined = target.value;

    if (target.type === 'number' || target.type === 'range') {
      value = target.value === '' ? undefined : Number(target.value);
      if (target.type === 'range') {
        this.updateSliderBackground(target);
      }
    } else if (target.type === 'checkbox') {
      value = target.checked;
    } else if (target.type === 'select-one') {
      const selectElement = target as unknown as HTMLSelectElement; 
      if (selectElement.options[selectElement.selectedIndex]?.disabled) {
        value = undefined; 
      } else {
        value = target.value;
      }
    }

    const newConfig = { ...this.config };

    if (key === 'auto-density') {
      this.autoDensity = Boolean(value);
      newConfig.density = this.autoDensity ? undefined : (this.lastDefinedDensity ?? 0.5);
    } else if (key === 'auto-brightness') {
      this.autoBrightness = Boolean(value);
      newConfig.brightness = this.autoBrightness ? undefined : (this.lastDefinedBrightness ?? 0.5);
    } else if (key === 'density') {
        if (value !== undefined) this.lastDefinedDensity = Number(value);
        if (!this.autoDensity) newConfig.density = Number(value);
    } else if (key === 'brightness') {
        if (value !== undefined) this.lastDefinedBrightness = Number(value);
        if (!this.autoBrightness) newConfig.brightness = Number(value);
    } else {
      (newConfig as any)[key] = value;
    }
    
    this.config = newConfig;
    this.dispatchSettingsChange();
  }

  override updated(changedProperties: Map<string | symbol, unknown>) {
    super.updated(changedProperties);
    if (changedProperties.has('config') || changedProperties.has('autoDensity') || changedProperties.has('autoBrightness')) {
      this.shadowRoot
        ?.querySelectorAll<HTMLInputElement>('input[type="range"]')
        .forEach((slider: HTMLInputElement) => {
          const key = slider.id as keyof LiveMusicGenerationConfig;
          let configValue: number | undefined;

          if (key === 'density') {
            configValue = this.autoDensity ? (this.lastDefinedDensity ?? 0.5) : this.config.density;
             slider.value = String(configValue ?? 0.5);
          } else if (key === 'brightness') {
            configValue = this.autoBrightness ? (this.lastDefinedBrightness ?? 0.5) : this.config.brightness;
            slider.value = String(configValue ?? 0.5);
          } else {
            configValue = this.config[key] as number | undefined;
            if (typeof configValue === 'number') {
                slider.value = String(configValue);
            }
          }
          if (slider.type === 'range') this.updateSliderBackground(slider);
        });
    }
  }

  private dispatchSettingsChange() {
    const configToSend = {...this.config};
    if (this.autoDensity) configToSend.density = undefined;
    if (this.autoBrightness) configToSend.brightness = undefined;

    this.dispatchEvent(
      new CustomEvent<LiveMusicGenerationConfig>('settings-changed', {
        detail: configToSend,
        bubbles: true,
        composed: true,
      }),
    );
  }

  private toggleAdvancedSettings() {
    this.showAdvanced = !this.showAdvanced;
  }

  override render() {
    const cfg = this.config;
    const advancedClasses = classMap({
      'advanced-settings': true,
      'visible': this.showAdvanced,
    });
    const scaleMap = new Map<string, string>([
      ['Auto', 'SCALE_UNSPECIFIED'],
      ['C Major / A Minor', 'C_MAJOR_A_MINOR'],
      ['C# Major / A# Minor', 'D_FLAT_MAJOR_B_FLAT_MINOR'],
      ['D Major / B Minor', 'D_MAJOR_B_MINOR'],
      ['D# Major / C Minor', 'E_FLAT_MAJOR_C_MINOR'],
      ['E Major / C# Minor', 'E_MAJOR_D_FLAT_MINOR'],
      ['F Major / D Minor', 'F_MAJOR_D_MINOR'],
      ['F# Major / D# Minor', 'G_FLAT_MAJOR_E_FLAT_MINOR'],
      ['G Major / E Minor', 'G_MAJOR_E_MINOR'],
      ['G# Major / F Minor', 'A_FLAT_MAJOR_F_MINOR'],
      ['A Major / F# Minor', 'A_MAJOR_G_FLAT_MINOR'],
      ['A# Major / G Minor', 'B_FLAT_MAJOR_G_MINOR'],
      ['B Major / G# Minor', 'B_MAJOR_A_FLAT_MINOR'],
    ]);

    return html`
      <div class="core-settings-row">
        <div class="setting">
          <label for="temperature"
            >Temp<span>${(cfg.temperature ?? this.defaultConfig.temperature!).toFixed(1)}</span></label
          >
          <input
            type="range"
            id="temperature"
            min="0"
            max="3"
            step="0.1"
            .value=${(cfg.temperature ?? this.defaultConfig.temperature!).toString()}
            @input=${this.handleInputChange} />
        </div>
        <div class="setting">
          <label for="guidance"
            >Guidance<span>${(cfg.guidance ?? this.defaultConfig.guidance!).toFixed(1)}</span></label
          >
          <input
            type="range"
            id="guidance"
            min="0"
            max="6"
            step="0.1"
            .value=${(cfg.guidance ?? this.defaultConfig.guidance!).toString()}
            @input=${this.handleInputChange} />
        </div>
        <div class="setting">
          <label for="topK">Top K<span>${cfg.topK ?? this.defaultConfig.topK!}</span></label>
          <input
            type="range"
            id="topK"
            min="1"
            max="100"
            step="1"
            .value=${(cfg.topK ?? this.defaultConfig.topK!).toString()}
            @input=${this.handleInputChange} />
        </div>
      </div>
      <hr class="divider" />
      <div class=${advancedClasses}>
        <div class="setting">
          <label for="seed">Seed</label>
          <input
            type="number"
            id="seed"
            .value=${cfg.seed?.toString() ?? ''}
            @input=${this.handleInputChange}
            placeholder="Auto" />
        </div>
        <div class="setting">
          <label for="bpm">BPM</label>
          <input
            type="number"
            id="bpm"
            min="60"
            max="180"
            .value=${cfg.bpm?.toString() ?? ''}
            @input=${this.handleInputChange}
            placeholder="Auto" />
        </div>
        <div class="setting" .auto=${this.autoDensity}>
          <label for="density">Density</label>
          <input
            type="range"
            id="density"
            min="0"
            max="1"
            step="0.05"
            .value=${String(this.autoDensity ? (this.lastDefinedDensity ?? 0.5) : (cfg.density ?? 0.5))}
            ?disabled=${this.autoDensity}
            @input=${this.handleInputChange} />
          <div class="auto-row">
            <input
              type="checkbox"
              id="auto-density"
              .checked=${this.autoDensity}
              @input=${this.handleInputChange} />
            <label for="auto-density">Auto</label>
            <span>${(this.autoDensity ? (this.lastDefinedDensity ?? 0.5) : (cfg.density ?? 0.5)).toFixed(2)}</span>
          </div>
        </div>
        <div class="setting" .auto=${this.autoBrightness}>
          <label for="brightness">Brightness</label>
          <input
            type="range"
            id="brightness"
            min="0"
            max="1"
            step="0.05"
            .value=${String(this.autoBrightness ? (this.lastDefinedBrightness ?? 0.5) : (cfg.brightness ?? 0.5))}
            ?disabled=${this.autoBrightness}
            @input=${this.handleInputChange} />
          <div class="auto-row">
            <input
              type="checkbox"
              id="auto-brightness"
              .checked=${this.autoBrightness}
              @input=${this.handleInputChange} />
            <label for="auto-brightness">Auto</label>
            <span>${(this.autoBrightness ? (this.lastDefinedBrightness ?? 0.5) : (cfg.brightness ?? 0.5)).toFixed(2)}</span>
          </div>
        </div>
        <div class="setting">
          <label for="scale">Scale</label>
          <select
            id="scale"
            .value=${cfg.scale || 'SCALE_UNSPECIFIED'}
            @change=${this.handleInputChange}>
            <option value="SCALE_UNSPECIFIED" ?selected=${!cfg.scale || cfg.scale === 'SCALE_UNSPECIFIED'}>Auto</option>
            ${[...scaleMap.entries()].filter(([_,val]) => val !== 'SCALE_UNSPECIFIED').map(
              ([displayName, enumValue]) =>
                html`<option value=${enumValue}>${displayName}</option>`,
            )}
          </select>
        </div>
        <div class="setting">
          <div class="setting checkbox-setting">
            <input
              type="checkbox"
              id="muteBass"
              .checked=${!!cfg.muteBass}
              @change=${this.handleInputChange} />
            <label for="muteBass" style="font-weight: normal;">Mute Bass</label>
          </div>
          <div class="setting checkbox-setting">
            <input
              type="checkbox"
              id="muteDrums"
              .checked=${!!cfg.muteDrums}
              @change=${this.handleInputChange} />
            <label for="muteDrums" style="font-weight: normal;"
              >Mute Drums</label
            >
          </div>
          <div class="setting checkbox-setting">
            <input
              type="checkbox"
              id="onlyBassAndDrums"
              .checked=${!!cfg.onlyBassAndDrums}
              @change=${this.handleInputChange} />
            <label for="onlyBassAndDrums" style="font-weight: normal;"
              >Only Bass & Drums</label
            >
          </div>
        </div>
      </div>
      <div class="advanced-toggle" @click=${this.toggleAdvancedSettings}>
        ${this.showAdvanced ? 'Hide' : 'Show'} Advanced Settings
      </div>
    `;
  }
}

// -----------------------------------------------------------------------------
// SECTION: Core Application Components (PromptDJ)
// -----------------------------------------------------------------------------

@customElement('prompt-dj')
class PromptDj extends LitElement {
  static override styles = css`
    :host {
      display: flex; 
      flex-direction: column;
      width: 100%;
      box-sizing: border-box;
      position: relative;
      font-size: 1.6vmin; 
      background-color: #222; 
      padding: 1vmin;
      border-radius: 4px;
    }
    .prompts-area {
      display: flex;
      align-items: flex-start; 
      justify-content: center;
      width: 100%;
      min-height: 10vmin; 
      margin-bottom: 1vmin; 
      gap: 1vmin;
      overflow-x: auto; 
      padding-bottom: 0.5vmin; 
      scrollbar-width: thin;
      scrollbar-color: #555 #333;
    }
    .prompts-area::-webkit-scrollbar { height: 6px; }
    .prompts-area::-webkit-scrollbar-track { background: #333; }
    .prompts-area::-webkit-scrollbar-thumb { background-color: #555; border-radius: 3px; }

    .prompt-text-display { 
        flex-grow: 1;
        padding: 0.5vmin;
        background-color: #1e1e1e;
        border-radius: 3px;
        max-height: 10vmin; 
        overflow-y: auto;
         scrollbar-width: thin;
        scrollbar-color: #555 #333;
    }
    .prompt-text-display div { 
        margin-bottom: 0.3em; 
        font-size: 1.3vmin;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

     .add-prompt-button-container { flex-shrink: 0; }
     #prompts-container {
        display: flex;
        flex-direction: row;
        align-items: flex-end; 
        flex-shrink: 1;
        height: 100%;
        gap: 1vmin;
        padding: 0.5vmin; 
        overflow-x: auto;
        max-width: 50%; 
    }


    #settings-container {
      width: 100%;
      margin-top: 1vmin;
    }
    .playback-container {
      display: flex;
      justify-content: center;
      align-items: center;
      margin-top: 1vmin;
      gap: 1vmin;
    }
    play-pause-button,
    reset-button { 
      width: 10vmin; 
      flex-shrink: 0;
    }
    add-prompt-button {
         width: 10vmin; flex-shrink: 0;
    }
    prompt-controller {
      height: 15vmin; 
      max-height: 20vmin;
      min-width: 12vmin;
      max-width: 15vmin; 
    }
    .background-vis { 
        height: 3px;
        background: linear-gradient(90deg, #9900ff, #5200ff, #ff25f6, #2af6de);
        margin-bottom: 1vmin;
        border-radius: 2px;
        opacity: 0.5;
    }
  `;

  @state() private userPrompts: Map<string, Prompt> = new Map();
  private nextUserPromptId: number = 0;
  
  private session!: LiveMusicSession; 
  private readonly sampleRate = 48000;
  private audioContext = new (window.AudioContext || (window as any).webkitAudioContext)( 
    {sampleRate: this.sampleRate},
  );
  private outputNode: GainNode = this.audioContext.createGain();
  private nextStartTime = 0;
  private readonly bufferTime = 2; 
  @state() private playbackState: PlaybackState = 'stopped';
  @property({type: Object}) private filteredPrompts = new Set<string>();
  private connectionError = true;

  @query('settings-controller') private settingsController!: SettingsController;
  
  @property({type: Object}) mcp!: ModelContextProtocol; 
  @state() private currentLyriaPrompts: WeightedPrompt[] = [];


  constructor() {
    super();
    this.outputNode.connect(this.audioContext.destination);
    this.userPrompts = this.getStoredUserPrompts();
    let maxNumericId = -1;
    for (const key of this.userPrompts.keys()) {
        if (key.startsWith('userprompt-')) {
            const num = parseInt(key.substring('userprompt-'.length), 10);
            if (!isNaN(num) && num > maxNumericId) maxNumericId = num;
        }
    }
    this.nextUserPromptId = maxNumericId + 1;
  }

  override async firstUpdated() {
    await this.connectToSession();
    if (this.mcp && this.mcp.lyria_output.current_prompt) {
        this.updateLyriaPrompts(this.mcp.lyria_output.current_prompt);
    }
  }

  public updateLyriaPrompts(promptAndConfig: MCPPromptAndConfig) {
    this.currentLyriaPrompts = promptAndConfig.weightedPrompts;
    this.setSessionPrompts(promptAndConfig.weightedPrompts);
    if (this.settingsController && promptAndConfig.config) {
      this.settingsController.setConfig(promptAndConfig.config); 
      this.updateLyriaConfig(promptAndConfig.config); 
    }
    this.requestUpdate(); 
  }

  public playAudio() {
      if (this.playbackState !== 'playing' && this.playbackState !== 'loading') {
          this.handlePlayPause();
      }
  }
  public pauseAudio() {
      if (this.playbackState === 'playing' || this.playbackState === 'loading') {
          this.internalPauseAudio();
      }
  }


  private async connectToSession() {
    try {
        this.session = await lyriaAI.live.music.connect({
        model: lyriaModel,
        callbacks: {
            onmessage: async (e: LiveMusicServerMessage) => {
            if (e.setupComplete) this.connectionError = false;
            if (e.filteredPrompt) {
                this.filteredPrompts = new Set([...this.filteredPrompts, e.filteredPrompt.text]);
                this.dispatchToast(`Filtered: ${e.filteredPrompt.filteredReason.substring(0,50)}`);
            }
            if (e.serverContent?.audioChunks !== undefined) {
                if (this.playbackState === 'paused' || this.playbackState === 'stopped') return;
                const audioBuffer = await decodeAudioData(decode(e.serverContent?.audioChunks[0].data), this.audioContext, 48000, 2);
                const source = this.audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(this.outputNode);
                if (this.nextStartTime === 0) {
                    this.nextStartTime = this.audioContext.currentTime + this.bufferTime;
                    setTimeout(() => { if(this.playbackState === 'loading') this.playbackState = 'playing'; }, this.bufferTime * 1000);
                }
                if (this.nextStartTime < this.audioContext.currentTime) {
                    console.log('Lyria audio under-run'); this.playbackState = 'loading'; this.nextStartTime = 0; return;
                }
                source.start(this.nextStartTime);
                this.nextStartTime += audioBuffer.duration;
            }
            },
            onerror: (errEvent: ErrorEvent) => {
                console.error('Lyria connection error:', errEvent);
                this.connectionError = true; this.internalStopAudio();
                this.dispatchToast('Lyria connection error. Please restart.');
            },
            onclose: (closeEvent: CloseEvent) => {
                console.log('Lyria connection closed.', closeEvent);
                if (!closeEvent.wasClean) { 
                    this.connectionError = true; this.internalStopAudio();
                    this.dispatchToast('Lyria connection closed unexpectedly.');
                }
            },
        },
        });
    } catch (err) {
        console.error("Failed to connect to Lyria session:", err);
        this.dispatchToast(`Failed to connect to Lyria: ${(err as Error).message.substring(0,100)}`);
        this.connectionError = true; this.playbackState = 'stopped';
    }
  }

  private setSessionPrompts = throttle(async (prompts: WeightedPrompt[]) => {
    if (!this.session || this.connectionError) return;
    const userWPs: WeightedPrompt[] = Array.from(this.userPrompts.values())
        .filter(p => p.weight > 0 && !this.filteredPrompts.has(p.text))
        .map(p => ({text: p.text, weight: p.weight}));

    const allPrompts = [...prompts, ...userWPs];

    if (allPrompts.length === 0) { 
        allPrompts.push({text: "silence", weight: 0.01}); 
    }

    try {
      await this.session.setWeightedPrompts({ weightedPrompts: allPrompts });
    } catch (e) {
      this.dispatchToast(`Error setting prompts: ${(e as Error).message.substring(0,100)}`);
      this.internalPauseAudio();
    }
  }, 200);

  private updateLyriaConfig = throttle(async (config: Partial<LiveMusicGenerationConfig>) => {
      if (!this.session || this.connectionError) return;
      try {
        await this.session.setMusicGenerationConfig({ musicGenerationConfig: config });
      } catch (e) {
        this.dispatchToast(`Error setting Lyria config: ${(e as Error).message.substring(0,100)}`);
      }
  }, 200);


  private dispatchToast(message: string) {
    this.dispatchEvent(new CustomEvent('show-toast', {
        detail: { message },
        bubbles: true,
        composed: true
    }));
  }
  
  private handleUserPromptChanged(e: CustomEvent<Prompt>) {
    const {promptId, text, weight, color, isTextEditable} = e.detail;
    const prompt = this.userPrompts.get(promptId);
    if (!prompt) return;
    
    const updatedPrompt: Prompt = { ...prompt, text, weight, color, isTextEditable };
    this.userPrompts.set(promptId, updatedPrompt);
    this.userPrompts = new Map(this.userPrompts); 
    
    this.setSessionPrompts(this.currentLyriaPrompts); 
    this.storeUserPrompts();
  }

  private async handleAddUserPrompt() {
    const newPromptId = `userprompt-${this.nextUserPromptId++}`;
    const usedColors = [...this.userPrompts.values()].map(p => p.color);
    const newPrompt: Prompt = {
      promptId: newPromptId,
      text: PROMPT_TEXT_PRESETS[Math.floor(Math.random() * PROMPT_TEXT_PRESETS.length)],
      weight: 0, color: getUnusedRandomColor(usedColors), isTextEditable: true,
    };
    this.userPrompts.set(newPromptId, newPrompt);
    this.userPrompts = new Map(this.userPrompts);

    this.setSessionPrompts(this.currentLyriaPrompts);
    this.storeUserPrompts();
  }
  
  private handleUserPromptRemoved(e: CustomEvent<string>) {
    e.stopPropagation();
    const promptIdToRemove = e.detail;
    if (this.userPrompts.has(promptIdToRemove)) {
      this.userPrompts.delete(promptIdToRemove);
      this.userPrompts = new Map(this.userPrompts);
      this.setSessionPrompts(this.currentLyriaPrompts);
      this.storeUserPrompts();
    }
  }

  private getStoredUserPrompts(): Map<string, Prompt> {
    const stored = localStorage.getItem('userPrompts');
    if (stored) {
        try { return new Map(JSON.parse(stored)); } catch (e) { console.error("Failed to parse stored user prompts", e); }
    }
    return new Map();
  }
  private storeUserPrompts() {
    localStorage.setItem('userPrompts', JSON.stringify(Array.from(this.userPrompts.entries())));
  }


  private async handlePlayPause() {
    if (this.playbackState === 'playing') {
      this.internalPauseAudio();
    } else if (this.playbackState === 'paused' || this.playbackState === 'stopped') {
      if (this.connectionError) {
        await this.connectToSession();
        if (!this.connectionError) {
          if (this.mcp) this.updateLyriaPrompts(this.mcp.lyria_output.current_prompt); 
          else this.setSessionPrompts([]); 
          this.internalLoadAudio();
        }
      } else {
         this.internalLoadAudio();
      }
    } else if (this.playbackState === 'loading') {
      this.internalStopAudio();
    }
  }

  private internalPauseAudio() {
    if (!this.session || this.connectionError) return;
    this.session.pause();
    this.playbackState = 'paused';
    this.outputNode.gain.setValueAtTime(this.outputNode.gain.value, this.audioContext.currentTime); 
    this.outputNode.gain.linearRampToValueAtTime(0, this.audioContext.currentTime + 0.1);
    this.nextStartTime = 0;
  }

  private internalLoadAudio() {
    if (!this.session || this.connectionError) {
        this.playbackState = 'stopped'; 
        this.dispatchToast("Cannot play: Lyria session not available.");
        return;
    }
    this.audioContext.resume();
    this.session.play();
    this.playbackState = 'loading';
    this.outputNode.gain.setValueAtTime(this.outputNode.gain.value, this.audioContext.currentTime);
    this.outputNode.gain.linearRampToValueAtTime(1, this.audioContext.currentTime + 0.1);
  }

  private internalStopAudio() {
    if (!this.session) return; 
    this.session.stop(); 
    this.playbackState = 'stopped';
    this.nextStartTime = 0;
  }
  
  private handleSettingsChanged(e: CustomEvent<LiveMusicGenerationConfig>) {
      this.updateLyriaConfig(e.detail);
      this.dispatchEvent(new CustomEvent('user-lyria-config-override', {
          detail: e.detail,
          bubbles:true, composed: true
      }));
  }

  private async handleReset() {
    if (this.connectionError) {
      await this.connectToSession();
       if (this.connectionError) { 
        this.dispatchToast("Cannot reset: Lyria connection failed.");
        return;
      }
    }
    this.internalPauseAudio();
    this.session.resetContext();
    this.settingsController.resetToDefaults(); 
    this.dispatchEvent(new CustomEvent('reset-mcp-musical-defaults', {bubbles: true, composed: true}));
        
    setTimeout(() => {
        if (!this.connectionError) this.internalLoadAudio();
    } , 100);
  }

  override render() {
    return html`
      <div class="background-vis"></div>
      <div class="prompts-area">
        <div class="prompt-text-display">
            <strong>MCP Prompts:</strong>
            ${this.currentLyriaPrompts.length > 0 ? 
                this.currentLyriaPrompts.map(p => html`<div>${p.text} (w: ${p.weight.toFixed(2)})</div>`) :
                html`<div>No prompts from MCP.</div>`
            }
        </div>
        <div id="prompts-container" @prompt-removed=${this.handleUserPromptRemoved}>
            ${Array.from(this.userPrompts.values()).map(p => html`
                <prompt-controller
                    .promptId=${p.promptId}
                    ?filtered=${this.filteredPrompts.has(p.text)}
                    .text=${p.text}
                    .weight=${p.weight}
                    .color=${p.color}
                    .isTextEditable=${p.isTextEditable ?? true}
                    @prompt-changed=${this.handleUserPromptChanged}>
                </prompt-controller>
            `)}
        </div>
        <div class="add-prompt-button-container">
            <add-prompt-button @click=${this.handleAddUserPrompt} title="Add User Prompt"></add-prompt-button>
        </div>
      </div>
      
      <div id="settings-container">
        <settings-controller @settings-changed=${this.handleSettingsChanged}></settings-controller>
      </div>
      <div class="playback-container">
        <play-pause-button
          @click=${this.handlePlayPause}
          .playbackState=${this.playbackState}></play-pause-button>
        <reset-button @click=${this.handleReset}></reset-button>
      </div>
      `;
  }
}

// -----------------------------------------------------------------------------
// SECTION: Top-Level Application Orchestrator (MultiAgentSystemDashboard)
// -----------------------------------------------------------------------------

@customElement('multi-agent-system-dashboard')
class MultiAgentSystemDashboard extends LitElement {
    static override styles = css`
        :host {
            display: flex;
            flex-direction: column;
            height: 100vh; 
            width: 100vw; 
            box-sizing: border-box;
            background-color: #1a1a1a;
            color: #eee;
            font-family: 'Google Sans', sans-serif;
            font-size: 1.4vmin;
            padding: 1vmin;
            overflow: hidden; 
        }
        .dashboard-header {
            text-align: center;
            padding: 1vmin;
            font-size: 2vmin;
            font-weight: bold;
            color: #667eea; 
        }
        .main-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1vmin;
            background-color: #2a2a2a;
            border-radius: 5px;
            margin-bottom: 1vmin;
        }
        .main-controls button {
            padding: 0.8vmin 1.5vmin;
            border-radius: 4px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            font-size: 1.4vmin;
        }
        .main-controls button.start { background-color: #667eea; color: white; }
        .main-controls button.stop { background-color: #ef4444; color: white; }
        .main-controls button:disabled { background-color: #555; cursor: not-allowed; }
        .main-controls .status-display { display: flex; align-items: center; gap: 0.5vmin;}
        .status-indicator { width: 1vmin; height: 1vmin; border-radius: 50%; display: inline-block; }
        .status-ready { background-color: #22c55e; } 
        .status-working { background-color: #f59e0b; animation: pulse 1.5s infinite; } 
        .status-error { background-color: #ef4444; } 

        .dashboard-layout {
            display: flex;
            flex-grow: 1; 
            gap: 1vmin;
            overflow: hidden; 
        }

        .left-column, .right-column {
            display: flex;
            flex-direction: column;
            gap: 1vmin;
            overflow-y: auto; 
             scrollbar-width: thin;
             scrollbar-color: #444 #2a2a2a;
        }
        .left-column { flex: 1; }
        .right-column { flex: 2; }


        .agent-pipeline-display {
            display: flex;
            align-items: center;
            justify-content: space-around;
            padding: 1vmin;
            background-color: #2a2a2a;
            border-radius: 5px;
            margin-bottom: 1vmin; 
        }
        .agent-step { text-align: center; font-size: 1.2vmin; flex: 1; }
        .agent-step .icon { font-size: 1.5vmin; margin-bottom: 0.2vmin; }
        .agent-step .label { display: block; }
        .agent-step .status-text { font-size: 1vmin; color: #aaa; text-transform: capitalize; }

        .agent-step.status-ready .status-text { color: #8f8; }
        .agent-step.status-working .status-text { color: #ff8; animation: pulse 1.5s infinite; }
        .agent-step.status-success .status-text { color: #8f8; }
        .agent-step.status-error .status-text { color: #f88; }
        
        .agent-arrow { color: #667eea; font-size: 1.5vmin; flex-shrink: 0; }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(10vmin, 1fr));
            gap: 1vmin;
            background-color: #2a2a2a;
            padding: 1vmin;
            border-radius: 5px;
        }
        .metric-box {
            background-color: #333;
            padding: 1vmin;
            border-radius: 4px;
            text-align: center;
        }
        .metric-value { font-size: 1.8vmin; font-weight: bold; color: #fff; }
        .metric-label { font-size: 1vmin; color: #aaa; text-transform: uppercase;}
        
        .agent-outputs {
             display: grid;
             grid-template-columns: 1fr; 
             gap: 1vmin;
        }

        .mcp-display-container, .lyria-interaction-container {
            background-color: #2a2a2a;
            padding: 1vmin;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
        }
        .mcp-display-container h3, .lyria-interaction-container h3 {
             margin-top: 0; font-size: 1.6vmin; color: #bbb; border-bottom: 1px solid #444; padding-bottom: 0.5vmin; margin-bottom: 0.5vmin;
        }
        #mcpDisplay {
            background-color: #1e1e1e;
            color: #e0e0e0;
            padding: 1vmin;
            border-radius: 4px;
            overflow: auto; 
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 1.2vmin;
            flex-grow: 1; 
            min-height: 20vmin; 
        }
        
        prompt-dj { 
             width: 100%;
             flex-shrink: 0; 
        }
        
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    `;

    @state() private mcp!: ModelContextProtocol;
    @state() private simulationRunning = false;
    @state() private overallStatus: 'ready' | 'working' | 'error' = 'ready';
    @state() private updateIntervalMs = 4000; 

    private simulationIntervalId: number | null = null;
    private sessionTimerId: number | null = null;
    private sessionStartTime = 0;
    @state() private sessionDurationDisplay = "00:00";


    @query('prompt-dj') private promptDjComponent!: PromptDj;
    @query('toast-message') private toastMessageComponent!: ToastMessage;

    private lastSimulatedHR = 70;
    private lastSimulatedCadence = 0;

    constructor() {
        super();
        this.initializeMcp();
    }

    private initializeMcp() {
        this.mcp = {
            context_schema_version: "2.1.0", // Updated version
            session_id: crypto.randomUUID(),
            timestamp: new Date().toISOString(),
            performance_metrics: {
                total_cycles: 0, successful_cycles: 0, avg_cycle_time: 0, efficiency_score: 1.0
            },
            instructions: {
                primary_goal: "Mantenere utente in Zona HR 3 con alto engagement musicale",
                adaptive_strategy: "Regolare parametri musicali basandosi su feedback fisiologico e preferenze apprese",
                style_preferences: ["Trip Hop", "Drum and Bass", "Funk", "Minimal Techno", "Ambient", "Lo-Fi Hip Hop"],
                constraints: {
                    hr_target_zone: [130, 150], max_bpm_change_per_cycle: 15,
                    fatigue_threshold: 0.8, mood_adaptation_sensitivity: 0.7
                }
            },
            agents: {
                ingestion: { id: "agent_ingestion_v2.1", status: "ready", specialization: "Elaborazione dati fisiologici", performance: { success_rate: 1.0, avg_response_time: 0 }, last_execution: null },
                context: { id: "agent_context_v2.1", status: "ready", specialization: "Analisi situazionale e trend", performance: { success_rate: 1.0, avg_response_time: 0 }, last_execution: null },
                musical: { id: "agent_musical_v2.1", status: "ready", specialization: "Generazione prompts Lyria", performance: { success_rate: 1.0, avg_response_time: 0 }, last_execution: null },
                feedback: { id: "agent_feedback_v2.1", status: "ready", specialization: "Apprendimento e ottimizzazione", performance: { success_rate: 1.0, avg_response_time: 0 }, last_execution: null }
            },
            system_state: {
                raw_data: { hr: 70, cadence: 0, motion: 0.1 },
                processed_metrics: { heart_rate_bpm: 70, hr_zone: 1, cadence_spm: 0, user_fatigue_score: 0.0, session_minute: 0, current_mood: "neutral", trend_direction: "stable" },
                context_analysis: { workout_phase: "idle", energy_level: "low", adaptation_needed: false, risk_factors: [], opportunities: [] }
            },
            memory: {
                user_profile: { preferred_bpm_range: [100, 140], instrument_preferences: { bass: 0.9, drums: 0.8, keys: 0.7, pad: 0.6, synth: 0.75, vocals: 0.4 }, style_affinity: { "Trip Hop": 0.8, "Drum and Bass": 0.7, "Funk": 0.9 } },
                session_history: [],
                learned_patterns: { hr_response_to_bpm: {}, fatigue_indicators: [], effective_transitions: [] }
            },
            lyria_output: {
                current_prompt: { weightedPrompts: [{ text: "ambient startup", weight: 1.0 }], config: { temperature: 0.8, topK: 40, guidance: 3.0, density: 0.3, brightness: 0.5 } },
                generation_history: [], effectiveness_scores: []
            },
            feedback_loop: {
                pattern_detection: { identified_patterns: [], confidence_scores: [] },
                adaptation_suggestions: [], learning_insights: []
            }
        };
    }
    
    private simulateSmartwatchData(): MCPRawData {
        const hrChange = (Math.random() - 0.5) * 8; 
        this.lastSimulatedHR = Math.max(60, Math.min(185, this.lastSimulatedHR + hrChange + (this.simulationRunning ? 1 : -2) )); 
        
        let cadenceChange = 0;
        if (this.lastSimulatedHR > 100 && this.simulationRunning) {
            cadenceChange = (Math.random() - 0.4) * 10; 
            this.lastSimulatedCadence = Math.max(70, Math.min(190, this.lastSimulatedCadence + cadenceChange));
        } else if (this.simulationRunning) {
             this.lastSimulatedCadence = Math.max(0, this.lastSimulatedCadence + 5); 
        } else {
            this.lastSimulatedCadence = Math.max(0, this.lastSimulatedCadence - 10); 
        }

        return {
            hr: Math.round(this.lastSimulatedHR),
            cadence: Math.round(this.lastSimulatedCadence),
            motion: +(Math.random() * 0.8 + 0.2).toFixed(2) 
        };
    }

    // --- Agent Execution Functions (Simulated API Calls) ---
    private async executeIngestionAgent(input: IngestionAgentInput): Promise<IngestionAgentOutput> {
        await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 100)); // Simulate async work
        
        const { raw_data, current_session_minute } = input;
        let hrZone = 1;
        if (raw_data.hr > 110) hrZone = 2; if (raw_data.hr > 130) hrZone = 3; 
        if (raw_data.hr > 150) hrZone = 4; if (raw_data.hr > 170) hrZone = 5;
        
        const fatigueScore = Math.min(1.0, (current_session_minute * 0.04) + (hrZone * 0.05));
        const mood = raw_data.hr > 140 ? (fatigueScore > 0.6 ? 'fatigued' : 'energetic') : 
                     raw_data.hr > 110 ? 'focused' : 'calm';
        
        const processed_metrics: MCPProcessedMetrics = {
            heart_rate_bpm: raw_data.hr, 
            hr_zone: hrZone, 
            cadence_spm: raw_data.cadence,
            user_fatigue_score: +fatigueScore.toFixed(2),
            session_minute: current_session_minute,
            current_mood: mood,
            trend_direction: raw_data.hr > (this.mcp.system_state.processed_metrics.heart_rate_bpm || raw_data.hr) + 3 ? 'increasing' 
                           : raw_data.hr < (this.mcp.system_state.processed_metrics.heart_rate_bpm || raw_data.hr) - 3 ? 'decreasing' : 'stable'
        };
        const reasoning = `HR: ${raw_data.hr}bpm (Zona ${hrZone}), Cad: ${raw_data.cadence}spm. Fatica: ${fatigueScore.toFixed(2)}. Mood: ${mood}.`;
        return { processed_metrics, reasoning, output_summary: processed_metrics };
    }

    private async executeContextAgent(input: ContextAgentInput): Promise<ContextAgentOutput> {
        await new Promise(resolve => setTimeout(resolve, 400 + Math.random() * 100));
        const { processed_metrics, fatigue_threshold, simulation_running } = input;
        let phase: MCPContextAnalysis['workout_phase'] = 'idle';
        if (simulation_running) {
           phase = processed_metrics.session_minute < 2 ? 'warmup' : 
                   processed_metrics.session_minute > 15 ? 'cooldown' : 'main';
        }
        const energy: MCPContextAnalysis['energy_level'] = processed_metrics.hr_zone > 3 ? 'high' : 
                                                            processed_metrics.hr_zone > 1 ? 'moderate' : 'low';
        const risks = processed_metrics.user_fatigue_score > fatigue_threshold ? ['high_fatigue'] : [];
        if (processed_metrics.hr_zone > 4 && phase === 'main') risks.push('overexertion_risk');
        const opportunities = processed_metrics.current_mood === 'energetic' && energy === 'moderate' ? ['increase_intensity'] : [];

        const context_analysis: MCPContextAnalysis = { 
            workout_phase: phase, energy_level: energy, 
            adaptation_needed: risks.length > 0 || opportunities.length > 0, 
            risk_factors: risks, opportunities: opportunities 
        };
        const reasoning = `Fase: ${phase}, Energia: ${energy}. Rischi: ${risks.join(',') || 'nessuno'}. Opportunità: ${opportunities.join(',') || 'nessuna'}.`;
        return { context_analysis, reasoning, output_summary: context_analysis };
    }

    private async executeMusicalAgent(input: MusicalAgentInput): Promise<MusicalAgentOutput> {
        await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 150));
        const { context_analysis: ctx, processed_metrics: procMetrics, style_preferences, total_cycles } = input;
        
        let basePrompt = "ambient flow";
        let targetBpm = procMetrics.heart_rate_bpm * 0.9; 
        let density = 0.4, brightness = 0.5;

        if (ctx.workout_phase === 'warmup') { basePrompt = "gentle evolving pads"; targetBpm = 100; density = 0.3; }
        else if (ctx.workout_phase === 'main') {
            basePrompt = procMetrics.current_mood === 'energetic' ? "driving electronic beat" : "focused rhythmic pulse";
            targetBpm = procMetrics.hr_zone > 3 ? 140 : 120;
            density = procMetrics.hr_zone > 2 ? 0.6 : 0.4;
            brightness = procMetrics.hr_zone > 2 ? 0.7 : 0.5;
        } else if (ctx.workout_phase === 'cooldown') { basePrompt = "calming soundscape"; targetBpm = 90; density = 0.2; }

        if (ctx.opportunities.includes('increase_intensity')) { basePrompt += ", energetic arps"; density = Math.min(0.8, density + 0.2); }
        if (ctx.risk_factors.includes('high_fatigue')) { basePrompt += ", minimalist"; density = Math.max(0.1, density - 0.2); brightness = Math.max(0.2, brightness - 0.2); }
        
        const newWeightedPrompts: WeightedPrompt[] = [{text: `${basePrompt}, ${procMetrics.current_mood} mood`, weight: 1.0 }];
        if (style_preferences.length > 0) {
            const style = style_preferences[total_cycles % style_preferences.length];
            newWeightedPrompts.push({text: style, weight: 0.7});
        }

        const lyria_prompt: MCPPromptAndConfig = {
            weightedPrompts: newWeightedPrompts,
            config: { 
                bpm: Math.round(targetBpm), 
                density: +density.toFixed(2), 
                brightness: +brightness.toFixed(2),
                temperature: 1.0, guidance: 3.5, topK: 50,
                muteDrums: ctx.risk_factors.includes('high_fatigue'),
            }
        };
        const reasoning = `Generato: ${basePrompt.substring(0,30)}... per ${ctx.workout_phase}, BPM ~${Math.round(targetBpm)}. Densità: ${density.toFixed(2)}.`;
        const output_summary = { prompt_summary: basePrompt.substring(0,30)+"...", config_summary: `BPM ${targetBpm.toFixed(0)}, D ${density.toFixed(2)}, B ${brightness.toFixed(2)}`};
        return { lyria_prompt, reasoning, output_summary };
    }

    private async executeFeedbackAgent(input: FeedbackAgentInput = {}): Promise<FeedbackAgentOutput> {
        await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 100));
        const reasoning = "Analisi output musicale vs contesto. (Apprendimento pattern non ancora implementato a fondo).";
        const output_summary = "Feedback loop attivo.";
        // In futuro, questo agente potrebbe restituire `updated_memory` per aggiornare MCP.
        return { reasoning, output_summary };
    }
    // --- End Agent Execution Functions ---

    private updateAgentState(agentKey: AgentKey, status: MCPAgent['status'], reasoning?: string, output_summary?: string | Record<string, any>) {
        const agent = this.mcp.agents[agentKey];
        this.mcp.agents[agentKey] = {
            ...agent,
            status: status,
            last_execution: status !== 'working' ? new Date().toISOString() : agent.last_execution, // Update on completion or error
            reasoning: reasoning ?? agent.reasoning,
            output_summary: output_summary ?? agent.output_summary,
        };
        // Forcing a full MCP update to ensure reactivity for nested agent objects
        this.mcp = {...this.mcp};
    }
    
    private handleAgentError(agentKey: AgentKey, error: unknown) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        this.updateAgentState(agentKey, 'error', `Errore: ${errorMessage}`);
        console.error(`Errore nell'agente ${agentKey}:`, error);
        this.overallStatus = 'error'; 
    }

    private async runSimulationCycle() {
        if (!this.simulationRunning) return;
        this.overallStatus = 'working';
        const cycleStartTime = performance.now();

        this.mcp.timestamp = new Date().toISOString();
        this.mcp.performance_metrics.total_cycles++;
        
        const currentSessionMinute = Math.floor((Date.now() - this.sessionStartTime) / 60000);
        this.mcp.system_state.processed_metrics.session_minute = currentSessionMinute;

        // 0. Simulate Smartwatch Data
        this.mcp.system_state.raw_data = this.simulateSmartwatchData();

        try {
            // 1. Ingestion Agent
            this.updateAgentState(AGENT_KEYS.INGESTION, 'working');
            const ingestionInput: IngestionAgentInput = { raw_data: this.mcp.system_state.raw_data, current_session_minute: currentSessionMinute };
            const ingestionOutput = await this.executeIngestionAgent(ingestionInput);
            this.mcp.system_state.processed_metrics = ingestionOutput.processed_metrics;
            this.updateAgentState(AGENT_KEYS.INGESTION, 'success', ingestionOutput.reasoning, ingestionOutput.output_summary);

            // 2. Context Agent
            this.updateAgentState(AGENT_KEYS.CONTEXT, 'working');
            const contextInput: ContextAgentInput = { 
                processed_metrics: this.mcp.system_state.processed_metrics, 
                fatigue_threshold: this.mcp.instructions.constraints.fatigue_threshold,
                simulation_running: this.simulationRunning
            };
            const contextOutput = await this.executeContextAgent(contextInput);
            this.mcp.system_state.context_analysis = contextOutput.context_analysis;
            this.updateAgentState(AGENT_KEYS.CONTEXT, 'success', contextOutput.reasoning, contextOutput.output_summary);

            // 3. Musical Agent
            this.updateAgentState(AGENT_KEYS.MUSICAL, 'working');
            const musicalInput: MusicalAgentInput = {
                context_analysis: this.mcp.system_state.context_analysis,
                processed_metrics: this.mcp.system_state.processed_metrics,
                style_preferences: this.mcp.instructions.style_preferences,
                total_cycles: this.mcp.performance_metrics.total_cycles
            };
            const musicalOutput = await this.executeMusicalAgent(musicalInput);
            this.mcp.lyria_output.current_prompt = musicalOutput.lyria_prompt;
            this.mcp.lyria_output.generation_history.unshift({...musicalOutput.lyria_prompt});
            if(this.mcp.lyria_output.generation_history.length > 10) this.mcp.lyria_output.generation_history.pop();
            
            this.mcp.memory.session_history.unshift({
                timestamp: new Date().toISOString(),
                metrics: JSON.parse(JSON.stringify(this.mcp.system_state.processed_metrics)),
                prompt_sent_to_lyria: JSON.parse(JSON.stringify(this.mcp.lyria_output.current_prompt))
            });
            if (this.mcp.memory.session_history.length > 20) this.mcp.memory.session_history.pop();
            this.updateAgentState(AGENT_KEYS.MUSICAL, 'success', musicalOutput.reasoning, musicalOutput.output_summary);
            
            if (this.promptDjComponent) {
                this.promptDjComponent.updateLyriaPrompts(this.mcp.lyria_output.current_prompt);
            }

            // 4. Feedback Agent
            this.updateAgentState(AGENT_KEYS.FEEDBACK, 'working');
            // const feedbackInput: FeedbackAgentInput = { /* ... relevant parts of MCP ... */ };
            const feedbackOutput = await this.executeFeedbackAgent({}); // Pass empty for now
            this.updateAgentState(AGENT_KEYS.FEEDBACK, 'success', feedbackOutput.reasoning, feedbackOutput.output_summary);
            // if (feedbackOutput.updated_memory) { /* Merge into this.mcp.memory */ }

            this.mcp.performance_metrics.successful_cycles++;
        } catch (error) {
            // Error already handled by handleAgentError within the try block if an agent throws
            console.error("Errore durante il ciclo di simulazione:", error);
            // Overall status already set to error by handleAgentError
        } finally {
            const cycleEndTime = performance.now();
            const cycleDuration = cycleEndTime - cycleStartTime;
            
            if (this.mcp.performance_metrics.successful_cycles > 0) {
              const totalTime = (this.mcp.performance_metrics.avg_cycle_time * (this.mcp.performance_metrics.total_cycles -1)) + cycleDuration; // use total_cycles for avg calculation
              this.mcp.performance_metrics.avg_cycle_time = totalTime / this.mcp.performance_metrics.total_cycles;
              this.mcp.performance_metrics.efficiency_score = this.mcp.performance_metrics.successful_cycles / this.mcp.performance_metrics.total_cycles;
            } else if (this.mcp.performance_metrics.total_cycles > 0) {
                 this.mcp.performance_metrics.efficiency_score = 0;
            }

            if (this.overallStatus === 'working') {
                this.overallStatus = 'ready';
            }
            this.mcp = {...this.mcp}; // Ensure MCP object reference changes for Lit
            this.requestUpdate(); 
        }
    }


    private startSimulation() {
        if (this.simulationRunning) return;
        this.simulationRunning = true;
        this.overallStatus = 'working';
        this.sessionStartTime = Date.now() - (this.mcp.system_state.processed_metrics.session_minute * 60000); 
        
        // Reset last simulated values for a fresh run feel
        this.lastSimulatedHR = this.mcp.system_state.raw_data.hr || 70;
        this.lastSimulatedCadence = this.mcp.system_state.raw_data.cadence || 0;

        this.simulationIntervalId = window.setInterval(() => {
            if (this.simulationRunning) this.runSimulationCycle();
        }, this.updateIntervalMs);
        
        this.sessionTimerId = window.setInterval(() => {
            if (this.simulationRunning) {
                const elapsedMs = Date.now() - this.sessionStartTime;
                const minutes = Math.floor(elapsedMs / 60000);
                const seconds = Math.floor((elapsedMs % 60000) / 1000);
                this.sessionDurationDisplay = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
            }
        }, 1000);

        this.runSimulationCycle();
        if (this.promptDjComponent) this.promptDjComponent.playAudio();
    }

    private stopSimulation() {
        this.simulationRunning = false;
        this.overallStatus = 'ready';
        if (this.simulationIntervalId) clearInterval(this.simulationIntervalId);
        this.simulationIntervalId = null;
        if (this.sessionTimerId) clearInterval(this.sessionTimerId);
        this.sessionTimerId = null;
        
        (Object.keys(this.mcp.agents) as Array<AgentKey>).forEach(key => {
            this.updateAgentState(key, 'ready');
        });
        if (this.promptDjComponent) this.promptDjComponent.pauseAudio();
        this.requestUpdate();
    }

    private handleIntervalChange(e: Event) {
        const input = e.target as HTMLInputElement;
        this.updateIntervalMs = Number(input.value); 
        if (this.simulationRunning) { 
            this.stopSimulation();
            this.startSimulation();
        }
    }

    private handleShowToast(e: CustomEvent<{message: string}>) {
        if (this.toastMessageComponent) {
            this.toastMessageComponent.show(e.detail.message);
        }
    }

    private handleUserLyriaConfigOverride(e: CustomEvent<LiveMusicGenerationConfig>) {
        console.log("User overridden Lyria config:", e.detail);
        this.mcp.lyria_output.current_prompt.config = {
            ...this.mcp.lyria_output.current_prompt.config,
            ...e.detail 
        };
        this.mcp = {...this.mcp};
        this.requestUpdate(); 
    }

    private handleResetMcpMusicalDefaults() {
        this.mcp.lyria_output.current_prompt = {
             weightedPrompts: [{ text: "ambient gentle reset", weight: 1.0 }],
             config: { temperature: 0.8, topK: 40, guidance: 3.0, density: 0.3, brightness: 0.5 }
        };
        if (this.toastMessageComponent) { 
            this.toastMessageComponent.show("MCP musical defaults were notionally reset.");
        }
        if (this.promptDjComponent) {
            this.promptDjComponent.updateLyriaPrompts(this.mcp.lyria_output.current_prompt);
        }
        this.mcp = {...this.mcp};
        this.requestUpdate();
    }

    override connectedCallback() {
        super.connectedCallback();
        this.addEventListener('show-toast', this.handleShowToast as EventListener);
        this.addEventListener('user-lyria-config-override', this.handleUserLyriaConfigOverride as EventListener);
        this.addEventListener('reset-mcp-musical-defaults', this.handleResetMcpMusicalDefaults as EventListener);
    }
    
    override disconnectedCallback() {
        super.disconnectedCallback();
        this.stopSimulation();
        this.removeEventListener('show-toast', this.handleShowToast as EventListener);
        this.removeEventListener('user-lyria-config-override', this.handleUserLyriaConfigOverride as EventListener);
        this.removeEventListener('reset-mcp-musical-defaults', this.handleResetMcpMusicalDefaults as EventListener);
    }

    renderAgentPipeline() {
        const agentDisplayData = [
            { key: AGENT_KEYS.INGESTION, icon: '🔄', label: 'Ingestione' },
            { key: AGENT_KEYS.CONTEXT, icon: '🧠', label: 'Contesto' },
            { key: AGENT_KEYS.MUSICAL, icon: '🎵', label: 'Musicale' },
            { key: AGENT_KEYS.FEEDBACK, icon: '📈', label: 'Feedback' }
        ];

        return html`
            <div class="agent-step">
                <div class="icon">📊</div>
                <div class="label">Smartwatch</div>
                <div class="status-text">Input</div>
            </div>
            <div class="agent-arrow">→</div>
            ${agentDisplayData.map((agentInfo, index) => html`
                <div class="agent-step status-${this.mcp.agents[agentInfo.key].status}">
                    <div class="icon">${agentInfo.icon}</div>
                    <div class="label">${agentInfo.label}</div>
                    <div class="status-text">${this.mcp.agents[agentInfo.key].status}</div>
                </div>
                ${index < agentDisplayData.length - 1 ? html`<div class="agent-arrow">→</div>` : ''}
            `)}
        `;
    }

    renderMetrics() {
        const metrics = this.mcp.system_state.processed_metrics;
        return html`
            <div class="metric-box">
                <div class="metric-value">${metrics.heart_rate_bpm}</div>
                <div class="metric-label">HR (BPM)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${metrics.hr_zone}</div>
                <div class="metric-label">HR Zone</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${metrics.cadence_spm}</div>
                <div class="metric-label">Cadenza</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${metrics.user_fatigue_score.toFixed(2)}</div>
                <div class="metric-label">Fatica</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${metrics.current_mood}</div>
                <div class="metric-label">Mood</div>
            </div>
             <div class="metric-box">
                <div class="metric-value">${this.sessionDurationDisplay}</div>
                <div class="metric-label">Tempo Sessione</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${this.mcp.performance_metrics.total_cycles}</div>
                <div class="metric-label">Cicli</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${(this.mcp.performance_metrics.efficiency_score * 100).toFixed(0)}%</div>
                <div class="metric-label">Efficienza</div>
            </div>
        `;
    }
    
    renderAgentCards() {
        return html`
            <mcp-agent-card .agent=${this.mcp.agents.ingestion} agentName="Ingestion" agentColor="#10b981"></mcp-agent-card>
            <mcp-agent-card .agent=${this.mcp.agents.context} agentName="Context" agentColor="#3b82f6"></mcp-agent-card>
            <mcp-agent-card .agent=${this.mcp.agents.musical} agentName="Musical" agentColor="#8b5cf6"></mcp-agent-card>
            <mcp-agent-card .agent=${this.mcp.agents.feedback} agentName="Feedback" agentColor="#f59e0b"></mcp-agent-card>
        `;
    }


    override render() {
        if (!this.mcp) return html`Inizializzazione MCP...`;
        
        const overallStatusClass = `status-${this.overallStatus}`;

        return html`
            <div class="dashboard-header">Lyria Real-Time Multi-Agent System</div>
            
            <div class="main-controls">
                <div>
                    <button class="start" @click=${this.startSimulation} ?disabled=${this.simulationRunning}>🚀 Avvia</button>
                    <button class="stop" @click=${this.stopSimulation} ?disabled=${!this.simulationRunning}>⏹️ Ferma</button>
                </div>
                <div>
                    <label for="updateInterval" style="margin-right:0.5vmin;">Intervallo (ms):</label>
                    <input type="number" id="updateInterval" .value=${this.updateIntervalMs.toString()} 
                           min="1000" max="10000" step="500" @change=${this.handleIntervalChange}
                           style="width: 6em; padding: 0.3vmin; background:#333; color:white; border:1px solid #555; border-radius:3px;">
                </div>
                <div class="status-display">
                    <span class="status-indicator ${overallStatusClass}"></span>
                    <span>Sistema ${this.overallStatus.charAt(0).toUpperCase() + this.overallStatus.slice(1)}</span>
                </div>
            </div>

            <div class="agent-pipeline-display">
                ${this.renderAgentPipeline()}
            </div>
            
            <div class="dashboard-layout">
                <div class="left-column">
                    <div class="metrics-grid">${this.renderMetrics()}</div>
                    <div class="lyria-interaction-container">
                        <h3>Lyria Player</h3>
                        <prompt-dj .mcp=${this.mcp}></prompt-dj>
                    </div>
                     <div class="mcp-display-container">
                        <h3>Smartwatch Raw Data</h3>
                        <pre id="mcpDisplay">${JSON.stringify(this.mcp.system_state.raw_data, null, 2)}</pre>
                    </div>
                </div>
                <div class="right-column">
                    <div class="agent-outputs">${this.renderAgentCards()}</div>
                    <div class="mcp-display-container" style="flex-grow:1;">
                        <h3>Model Context Protocol (MCP)</h3>
                        <pre id="mcpDisplay">${JSON.stringify(this.mcp, (key, value) => {
                            // Optionally truncate long arrays for display
                            if (Array.isArray(value) && value.length > 5 && (key === 'session_history' || key === 'generation_history')) {
                                return `[${value.length} items, e.g., ${JSON.stringify(value[0])}, ...]`;
                            }
                            return value;
                        } , 2)}</pre>
                    </div>
                </div>
            </div>
            <toast-message></toast-message> 
        `;
    }
}

// -----------------------------------------------------------------------------
// SECTION: Main Execution and Global Declarations
// -----------------------------------------------------------------------------

function main(container: HTMLElement) {
  const dashboard = new MultiAgentSystemDashboard();
  container.appendChild(dashboard);
}

main(document.body);

declare global {
  interface HTMLElementTagNameMap {
    'prompt-dj': PromptDj;
    'prompt-controller': PromptController;
    'settings-controller': SettingsController;
    'multi-agent-system-dashboard': MultiAgentSystemDashboard;
    'mcp-agent-card': MCPAgentCard;
    'add-prompt-button': AddPromptButton;
    'play-pause-button': PlayPauseButton;
    'reset-button': ResetButton;
    'weight-slider': WeightSlider;
    'toast-message': ToastMessage;
  }
}
interface Window {
    webkitAudioContext: typeof AudioContext;
}
