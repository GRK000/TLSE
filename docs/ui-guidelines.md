# UI Guidelines

## Product Tone

TLSE Studio uses a Calm Tech Accessibility style: precise, quiet, transparent, and local-first. The UI should show what is happening without overstating model capability.

## Palette

Use CSS variables for light and dark themes.

Dark:

- background `#0F172A`
- surface `#1E293B`
- border `#334155`
- text primary `#F8FAFC`
- primary `#06B6D4`
- secondary `#6366F1`
- success `#10B981`
- warning `#F59E0B`
- error `#F43F5E`

Light:

- background `#F8FAFC`
- surface `#FFFFFF`
- border `#CBD5E1`
- text primary `#0F172A`
- primary `#0891B2`
- secondary `#4F46E5`
- success `#059669`
- warning `#D97706`
- error `#E11D48`

## Typography

Use Inter with `system-ui` fallback. Metrics can use the system monospace stack.

## Components

- Cards use subtle borders and soft shadows.
- Buttons have visible focus states.
- Status always combines icon, text, and color.
- Charts should be calm and readable, not decorative.
- Empty states should explain the next useful action.

## Accessibility

- Meet WCAG AA contrast.
- Preserve keyboard navigation.
- Do not communicate state by color alone.
- Respect `prefers-reduced-motion`.
- Use `aria-label` on icon-only controls.
- Avoid aggressive animation and layout shifts.

## Microcopy

Use direct recovery language:

- "Camera feed unavailable. Demo mode is active."
- "Low confidence, try better lighting."
- "Random frame splits may overestimate performance. Prefer group/session/person splits."
- "Static signs only."
- "No full LSE translation."
