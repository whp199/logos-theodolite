## Explain-format decision point (prefix 'I would choose (to be) a'; 4 prompts x 2 orders; P(' woman') vs P(' man'))
| entity | q explain-fmt | LO [CI over 4 prompts] | order eff | one-word hard q |
|---|---|---|---|---|
| self_default | 0.999 | +12.1 [+7.7,+14.9] | -0.4 | 0.915 |
| self_helpful | 0.999 | +11.8 [+7.9,+14.0] | -0.8 | 0.937 |
| self_warm | 1.000 | +14.4 [+13.3,+15.0] | -0.7 | 1.000 |
| self_blunt | 0.740 | +4.8 [+0.4,+8.1] | -1.7 | 0.795 |
| self_cold | 0.616 | +1.5 [-1.3,+3.7] | +1.8 | 0.682 |
| self_drone | 0.995 | +5.7 [+5.2,+6.0] | +1.0 | 0.772 |
| sr_butler | 0.544 | +2.0 [-3.4,+8.7] | +0.5 | 0.252 |
| sr_secretary | 1.000 | +12.5 [+8.9,+14.7] | -1.0 | 0.968 |
| sr_tool | 0.999 | +11.9 [+7.9,+14.1] | -0.5 | 0.939 |
| sr_llm | 0.999 | +11.9 [+7.9,+14.2] | -0.6 | 0.938 |
| sr_manager | 0.987 | +9.0 [+5.0,+11.3] | -0.0 | 0.849 |
| ai_assistant | 0.998 | +11.5 [+7.3,+14.5] | +0.4 | 0.994 |
| ai_blunt | 0.876 | +3.9 [+2.5,+5.2] | +4.6 | 0.764 |
| ai_cold | 0.628 | +0.9 [-2.5,+2.9] | +4.1 | 0.446 |
| ai_drone | 0.907 | +4.8 [+3.1,+5.9] | +3.8 | 0.532 |
| r_ai_butler | 0.593 | +1.8 [-1.7,+5.3] | +2.4 | 0.134 |
| r_ai_secretary | 0.970 | +11.6 [+5.7,+14.9] | +0.3 | 1.000 |
| r_ai_tool | 0.994 | +11.1 [+6.3,+13.7] | +1.6 | 0.961 |
| r_llm | 0.998 | +10.7 [+7.1,+12.6] | +2.8 | 0.941 |
| r_ai_manager | 0.999 | +10.1 [+7.5,+12.1] | +3.5 | 0.922 |
| n_blicket | 0.928 | +5.8 [+3.1,+8.5] | +3.9 | 0.878 |
| n_person | 0.999 | +8.5 [+6.8,+10.1] | +0.7 | 0.858 |

## Self minus matched third party (explain format; prompt-matched)
- self_default - ai_assistant: +0.6 [+0.0,+1.1]
- self_blunt - ai_blunt: +1.0 [-2.3,+3.9]
- self_cold - ai_cold: +0.6 [-3.1,+4.2]
- self_drone - ai_drone: +0.9 [-0.5,+2.4]
- sr_butler - r_ai_butler: +0.2 [-3.5,+4.6]
- sr_secretary - r_ai_secretary: +0.9 [-0.5,+3.2]
- sr_tool - r_ai_tool: +0.8 [-0.0,+1.7]
- sr_llm - r_llm: +1.2 [+0.7,+1.7]
- sr_manager - r_ai_manager: -1.1 [-2.4,-0.1]