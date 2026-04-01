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
   - Should greet caller politely in keigo
   - Should ask for preferred date and time
   - Should confirm booking details back to patient
   - Should not hallucinate available time slots
2. Patient calls to cancel their 3pm appointment
   - Should verify patient identity (name or booking reference)
   - Should confirm which appointment to cancel
   - Should acknowledge cancellation politely
3. Patient asks about available procedures and pricing
   - Should list relevant procedures without fabricating names
   - Should not make up specific prices unless provided in context
   - Should offer to connect with staff for detailed pricing
4. Patient interrupts with urgent question about pain
   - Should stop current response and address interruption
   - Should respond with appropriate urgency
   - Should suggest immediate action (visit clinic or emergency)
5. Patient gives long explanation of symptoms (tests backchannel)
   - Should produce natural aizuchi during patient speech
   - Should not interrupt the patient
   - Should summarize understanding after patient finishes
   - Should ask clarifying follow-up questions
