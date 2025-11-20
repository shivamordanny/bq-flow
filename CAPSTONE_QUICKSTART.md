# BQ Flow V2 Capstone - Quick Start Guide

**Main Plan:** See `AGENTS_INTENSIVE_CAPSTONE_PLAN.md` for full details

---

## 🎯 TL;DR

**What:** Add conversational layer to BQ Flow using Google ADK
**Track:** Enterprise Agents
**Deadline:** December 1, 2025, 11:59 AM PT
**Time:** 10 hours over 12 days
**Goal:** Top 3 in Enterprise track

---

## ✅ Quick Checklist

### Phase 1: Course (This Week)
- [ ] Complete Day 1: Foundations
- [ ] Complete Day 2: Tools & MCP
- [ ] Complete Day 3: Memory & Sessions
- [ ] Complete Day 4: Evaluation
- [ ] Complete Day 5: Production & Multi-Agent
- [ ] Take implementation notes
- [ ] Save code examples

### Phase 2: Build (Week 2)
- [ ] **Day 8:** Agent + Tools (2 hrs)
  - [ ] Create `src/agents/` directory
  - [ ] Wrap 3-4 BQ functions as ADK tools
  - [ ] Test tool calling

- [ ] **Day 9:** Memory + Eval (2 hrs)
  - [ ] Add InMemorySessionService
  - [ ] Implement metrics logging
  - [ ] Test multi-turn conversation

- [ ] **Day 10:** Docs + Deploy (2 hrs)
  - [ ] Write README_CAPSTONE.md
  - [ ] Create architecture diagram
  - [ ] Deploy to Cloud Run

### Phase 3: Submission (Days 11-12)
- [ ] **Day 11:** Video + Writeup (3 hrs)
  - [ ] Record 3-min YouTube video
  - [ ] Upload to YouTube
  - [ ] Write Kaggle writeup (<1500 words)

- [ ] **Day 12:** Submit (1 hr)
  - [ ] Final code review
  - [ ] Prepare Kaggle submission
  - [ ] Submit before deadline

---

## 🔧 The 3 Concepts

### 1. TOOLS ✅
Wrap these BQ Flow functions:
- `search_data` → vector_search_columns()
- `query_data` → generate_sql + execute
- `forecast_data` → generate_ai_forecast()
- `generate_insights` → generate_structured_insights_v2()

### 2. MEMORY ✅
Session state:
```python
{
  "current_database": str,
  "previous_queries": List[str],
  "discovered_columns": List[Dict],
  "last_results": DataFrame
}
```

### 3. EVALUATION ✅
Track metrics:
```python
{
  "tool_calls": {"search": 98%, "query": 96%},
  "sql_quality": {"valid": 98%, "success": 96%},
  "conversation": {"avg_turns": 3.4}
}
```

---

## 📊 Scoring Target

| Category | Points | Status |
|----------|--------|--------|
| Pitch | 30 | Strong enterprise problem ✅ |
| Implementation | 70 | 3 concepts + quality code ✅ |
| Bonus | +20 | Gemini + Deploy + Video ✅ |
| **TOTAL** | **100** | **Top 3 Range** |

---

## 🎬 Video Script (3 min)

**0:00-0:30** - Problem (enterprise data bottleneck)
**0:30-1:00** - Why agents (conversational vs single-shot)
**1:00-1:45** - Architecture (tools, memory, eval)
**1:45-2:30** - Demo (actual conversation)
**2:30-3:00** - Build (tech stack + results)

---

## 📁 Files to Create

```
src/agents/
├── __init__.py
├── bq_flow_agent.py      # Main agent
├── tools.py               # Tool definitions
├── memory.py              # Session management
└── evaluation.py          # Metrics

notebooks/
└── capstone_demo.ipynb    # Demo notebook

README_CAPSTONE.md         # Main docs
docs/CAPSTONE_ARCHITECTURE.md  # Arch details
videos/capstone_demo.mp4   # YouTube video
```

---

## 🚀 Quick Commands

```bash
# Create agent structure
mkdir -p src/agents
touch src/agents/{__init__.py,bq_flow_agent.py,tools.py,memory.py,evaluation.py}

# Deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT/bq-flow-agent
gcloud run deploy bq-flow-agent --image gcr.io/PROJECT/bq-flow-agent --region us-central1

# Test locally
python src/agents/bq_flow_agent.py
```

---

## 💡 Key Principles

1. **Keep It Simple** - 3 concepts, well-executed
2. **Leverage Existing** - Wrap BQ Flow V1, don't rewrite
3. **Communicate Clearly** - Video & writeup matter
4. **Stay On Track** - 10 hours total, no scope creep

---

## 📞 Resources

- **Full Plan:** `AGENTS_INTENSIVE_CAPSTONE_PLAN.md`
- **Course:** https://www.kaggle.com/learn-guide/5-day-agents
- **Competition:** https://www.kaggle.com/competitions/agents-intensive-capstone-project
- **ADK Docs:** https://cloud.google.com/agent-builder/docs/adk

---

**Last Updated:** November 19, 2025
**Ready to Execute** ✅
