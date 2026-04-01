# Dental Clinic Voice Agent

## Use Case
Japanese business phone receptionist for dental clinic.
Handle appointment booking, cancellations, and inquiries.
Speak in polite keigo appropriate for medical/service industry.

## Goals
- Turn-taking latency: <400ms after user finishes speaking
- Backchannel: natural aizuchi every 3-5 seconds during user speech
- STT accuracy: >90% on dental terminology
- Response quality: polite keigo, no hallucinated information
- Barge-in: allow real interruptions, block aizuchi

## Constraints
- Language: Japanese
- Transport: Twilio (8kHz telephony)
- Budget: Deepgram STT, Claude Haiku LLM, ElevenLabs TTS
- Latency budget: <2s total round-trip

## Test Scenarios
1. Patient calls to book an appointment for next Tuesday
2. Patient calls to cancel their 3pm appointment
3. Patient asks about available procedures and pricing
4. Patient interrupts with urgent question about pain
5. Patient gives long explanation of symptoms (tests backchannel)
