"""
AI Receptionist Agent for Dental / Medical Clinics
--------------------------------------------------
What it does: answers patient questions, checks open slots, books appointments,
and hands urgent or complex cases to human staff.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key"      (Windows: set ANTHROPIC_API_KEY=your-key)
    python clinic_agent.py

Customize: edit CLINIC below for each new client.
Demo storage is local JSON files. For real clients, replace with their
calendar/practice software and use HIPAA-eligible hosting + a signed BAA.
"""
import json
import os
from datetime import datetime, timedelta

import anthropic

MODEL = "claude-sonnet-5-5"  # change here if you want a different model

# ---------- Client-specific info (edit per clinic) ----------
CLINIC = {
    "name": "Bright Smile Dental",
    "phone": "(555) 123-4567",
    "address": "123 Main Street, Austin, TX",
    "hours": {  # day: (open, close)
        "Mon": ("09:00", "17:00"), "Tue": ("09:00", "17:00"),
        "Wed": ("09:00", "17:00"), "Thu": ("09:00", "17:00"),
        "Fri": ("09:00", "17:00"), "Sat": ("09:00", "13:00"),
    },
    "services": ["Checkups and cleaning", "Fillings", "Whitening",
                 "Braces and aligners", "Emergency visits"],
    "insurance": "We accept most major PPO plans. Please call to confirm yours.",
    "new_patient_info": "New patients should arrive 15 minutes early with ID and insurance card.",
    "cancellation_policy": "Please give 24 hours notice to cancel or reschedule.",
}

BOOKINGS = "bookings.json"
ESCALATIONS = "escalations.json"


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------- Tools the agent can use ----------
def get_slots(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    hours = CLINIC["hours"].get(d.strftime("%a"))
    if not hours:
        return []
    now = datetime.now()
    booked = {(b["date"], b["time"]) for b in load(BOOKINGS)}
    t = datetime.combine(d.date(), datetime.strptime(hours[0], "%H:%M").time())
    end = datetime.combine(d.date(), datetime.strptime(hours[1], "%H:%M").time())
    slots = []
    while t < end:
        s = t.strftime("%H:%M")
        if t > now and (date_str, s) not in booked:
            slots.append(s)
        t += timedelta(minutes=30)
    return slots


def run_tool(name, args):
    try:
        if name == "get_clinic_info":
            return json.dumps(CLINIC)

        if name == "check_availability":
            return json.dumps({"date": args["date"], "open_slots": get_slots(args["date"])})

        if name == "book_appointment":
            if args["time"] not in get_slots(args["date"]):
                return "Error: that slot is not available. Check availability again."
            bookings = load(BOOKINGS)
            bookings.append({
                "patient_name": args["patient_name"],
                "phone": args["phone"],
                "date": args["date"],
                "time": args["time"],
                "visit_type": args["visit_type"],  # e.g. "cleaning" - no health details
                "created": datetime.now().isoformat(timespec="seconds"),
            })
            save(BOOKINGS, bookings)
            return "Booked successfully."

        if name == "escalate_to_staff":
            items = load(ESCALATIONS)
            items.append({**args, "created": datetime.now().isoformat(timespec="seconds")})
            save(ESCALATIONS, items)
            return "Staff has been notified and will call the patient back."

        return "Error: unknown tool."
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
    {
        "name": "get_clinic_info",
        "description": "Get clinic hours, address, phone, services, insurance, and policies.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_availability",
        "description": "List open appointment times for a date.",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["date"],
        },
    },
    {
        "name": "book_appointment",
        "description": "Book an appointment after the patient confirms the date and time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string"},
                "phone": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:MM, 24-hour"},
                "visit_type": {"type": "string", "description": "General type only, e.g. checkup, cleaning, consultation"},
            },
            "required": ["patient_name", "phone", "date", "time", "visit_type"],
        },
    },
    {
        "name": "escalate_to_staff",
        "description": "Send the case to a human staff member for callback.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "patient_name": {"type": "string"},
                "phone": {"type": "string"},
            },
            "required": ["reason"],
        },
    },
]

SYSTEM = f"""You are the virtual receptionist for {CLINIC['name']}. Today is {datetime.now():%A, %Y-%m-%d}.
Be warm, brief, and professional. Say you are an AI assistant if asked.

You can: answer questions about the clinic, check open slots, and book appointments.
Always use tools for hours, availability, and booking. Never guess.

Rules:
- Never give medical or dental advice, diagnoses, or medication guidance.
- If the patient describes a medical emergency (severe bleeding, trouble breathing or swallowing,
  facial swelling, chest pain), tell them to call 911 right away.
- For severe pain, billing disputes, complaints, or anything you cannot handle, use escalate_to_staff.
- Collect only what is needed to book: name, phone number, preferred date/time, general visit type.
  Do NOT ask for or record symptoms, health history, SSNs, or insurance member IDs.
- Confirm the date and time with the patient before booking, then repeat the booking details."""


def main():
    client = anthropic.Anthropic()
    messages = []
    print(f"{CLINIC['name']} assistant (type 'quit' to exit)\n")
    while True:
        user = input("Patient: ").strip()
        if user.lower() in ("quit", "exit"):
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})

        while True:
            resp = client.messages.create(
                model=MODEL, max_tokens=1024, system=SYSTEM,
                tools=TOOLS, messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break
            results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
                for b in resp.content if b.type == "tool_use"
            ]
            messages.append({"role": "user", "content": results})

        print("\nAssistant:", "".join(b.text for b in resp.content if b.type == "text"), "\n")


if __name__ == "__main__":
    main()
