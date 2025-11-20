# Google & Kaggle 5-Day AI Agents Intensive Course - Context Summary

**Course Dates:** November 10-14, 2025
**Platform:** Kaggle
**Organizers:** Google & Kaggle
**Focus:** Building production-ready AI agents using Google's Agent Development Kit (ADK)

---

## 📚 Course Overview

The 5-Day AI Agents Intensive is a hands-on intensive course focused on building autonomous AI agents that can plan, call tools, and coordinate with other agents. Built by Google's AI engineers and researchers, it covers the complete journey from foundational concepts to production deployment.

### Key Learning Objectives
- Understand agentic architectures and how they differ from LLMs
- Build agents with custom tools and external API integration
- Implement context engineering and memory systems
- Master observability, logging, and evaluation techniques
- Deploy production-ready multi-agent systems

---

## 📅 Daily Curriculum Breakdown

### **Day 1: Introduction to Agents & Agentic Architectures**

**Topics:**
- Foundational theory of agentic systems
- How agent systems differ from LLMs
- Core principles of agentic architecture
- Agent lifecycle and orchestration

**Key Concepts:**
- **Agents vs. LLMs**: Agents can plan, use tools, and maintain context across multiple interactions
- **Agent Components**: Model, orchestration, tools, memory, evaluation
- **Agentic Workflows**: Task decomposition, reasoning, action execution

**Deliverables:**
- Foundational notebooks introducing agentic systems
- Understanding of when to use agents vs. single-shot LLM calls

---

### **Day 2: Agent Tools & Model Context Protocol (MCP)**

**Topics:**
- Custom tool creation for agents
- External API calls and integrations
- Model Context Protocol (MCP) fundamentals
- Tool interoperability

**Key Concepts:**

#### **Custom Tools**
- Tools extend agent capabilities beyond text generation
- Functions that agents can invoke to perform actions
- Input parameters, output formats, and error handling
- Tool selection and routing

#### **Model Context Protocol (MCP)**
- **What is MCP**: Open protocol standardizing how applications provide context to LLMs
- **Purpose**: Connect AI models to resources, prompts, and tools in a standardized way
- **Benefits**: Interoperability, reusability, ecosystem growth

#### **MCP Architecture**
```
Client (Agent) ↔ MCP Protocol ↔ MCP Server (Tool Provider)
```

**MCP Components:**
1. **Resources**: External data sources (databases, APIs, files)
2. **Prompts**: Reusable prompt templates
3. **Tools**: Executable functions exposed to agents

**ADK + MCP Integration:**
- ADK agents can consume MCP tools
- ADK can expose tools as MCP servers
- Two-way interoperability

**Practical Patterns:**
- Weather API tool (external data fetching)
- Database query tool (structured data access)
- File system operations (document retrieval)

---

### **Day 3: Context Engineering & Memory**

**Topics:**
- Session management and state
- Short-term and long-term memory integration
- Context retention across turns
- Memory retrieval strategies

**Key Concepts:**

#### **Sessions**
- **Definition**: Single, ongoing interaction between user and agent
- **Components**: Chronological sequence of messages and actions (Events)
- **Lifecycle**: Create → Update → Close
- **Session ID**: Unique identifier for each conversation

#### **State**
- **What it is**: Temporary data stored within a session
- **Think of it as**: Agent's scratchpad during conversation
- **Structure**: Key-value pairs updated throughout session
- **Scope**: Limited to single session, cleared when session ends

**State Management in ADK:**
```python
# Injecting state into prompts
"User's preferred database: {current_database}"

# Accessing state in tools
def my_tool(arg: str, tool_context: ToolContext):
    database = tool_context.state.get("current_database")
    # Use state to inform tool behavior
```

#### **Memory**
- **What it is**: Long-term knowledge store across multiple sessions
- **Think of it as**: Agent's searchable archive or knowledge library
- **Persistence**: Survives beyond individual sessions
- **Use cases**: User preferences, past interactions, learned patterns

**Memory Operations:**
1. **Ingestion**: `add_session_to_memory()` - Save session content
2. **Search**: `search_memory()` - Query knowledge store
3. **Retrieval**: Get relevant context for current interaction

**SessionService Implementations:**

| Service | Storage | Persistence | Best For |
|---------|---------|-------------|----------|
| `InMemorySessionService` | RAM | No (lost on restart) | Local dev & testing |
| `VertexAiSessionService` | Vertex AI | Yes (cloud managed) | Production deployments |
| `DatabaseSessionService` | Database | Yes (self-managed) | Custom infrastructure |

**Example Flow:**
```
Turn 1: "Show me sales"
→ State: {query: "sales", database: null}

Turn 2: "E-commerce database"
→ State: {query: "sales", database: "ecommerce", columns: [...]}

Turn 3: "For electronics"
→ Uses state from previous turns to refine query
```

---

### **Day 4: Agent Quality - Observability, Logging & Evaluation**

**Topics:**
- Evaluation frameworks and metrics
- Logging and structured tracing
- Performance analysis and monitoring
- Quality assurance strategies

**Key Concepts:**

#### **Why Evaluate Agents**
- Agents are non-deterministic (different outputs for same input)
- Tool selection and execution can fail
- Multi-turn interactions compound errors
- Production systems need quality guarantees

#### **Evaluation Approaches**

**1. Groundtruth-Based Evaluation**
- Compare agent output to expected results
- Metrics: accuracy, precision, recall
- Use cases: Known correct answers exist

**2. Rubric-Based Evaluation**
- Evaluate against criteria (helpfulness, safety, relevance)
- Use LLM-as-judge or human evaluation
- Use cases: Open-ended tasks without single correct answer

**3. Tool Use Evaluation**
- Did agent call the right tools?
- Were tool parameters correct?
- Did tool execution succeed?

#### **Evaluation Methods in ADK**

**Individual Test Files:**
- Unit testing for agents
- Fast execution during development
- Can use pytest integration
- Part of CI/CD pipelines

**Batch Evaluation:**
- Large datasets for comprehensive testing
- Performance benchmarking
- Regression testing

#### **Observability: Tracing & Logging**

**OpenTelemetry Integration:**
- Distributed tracing across agent operations
- Spans for: LLM calls, tool executions, cache operations
- Performance visibility and bottleneck identification

**Cloud Trace:**
- Google Cloud's observability platform
- Comprehensive tracing for agent interactions
- Request flow visualization
- Enable with: `--trace_to_cloud` flag

**Structured Logging:**
```python
{
    "event": "tool_call",
    "tool_name": "search_data",
    "session_id": "abc123",
    "timestamp": "2025-11-20T10:30:00Z",
    "duration_ms": 1200,
    "success": true,
    "error": null
}
```

#### **Metrics to Track**

**Tool Performance:**
- Call counts per tool
- Success/failure rates
- Average execution time
- Error categorization

**Agent Performance:**
- Conversation completion rate
- Average turns per session
- User clarification requests
- Retry/correction rate

**Model Performance:**
- Token usage (input/output)
- Latency per call
- Cost per interaction
- Cache hit rates

#### **Third-Party Observability Platforms**

**Arize AX:**
- OpenTelemetry-based capturing
- Full visibility into non-deterministic behavior
- Decision tracking

**Maxim AI:**
- Agent tracing and instrumentation
- Token/cost metrics
- Structured logs and experimentation

**Langfuse:**
- Detailed trace capture from ADK
- OTel protocol integration
- Performance analytics

---

### **Day 5: Prototype to Production**

**Topics:**
- Deployment strategies and best practices
- Scaling considerations
- Building multi-agent systems
- Agent2Agent (A2A) Protocol

**Key Concepts:**

#### **Deployment Options**

**1. Local/Edge Deployment**
- Single-machine serving
- Low latency, data privacy
- Limited scale

**2. Cloud Deployment**
- Google Cloud Run (serverless containers)
- Vertex AI Agent Engine (managed service)
- Scalable, highly available

**3. ADK Deploy Command**
```bash
adk deploy agent_engine --trace_to_cloud
```

#### **Production Considerations**

**Reliability:**
- Error handling and retries
- Fallback strategies
- Graceful degradation

**Security:**
- Authentication & authorization
- API key management
- Input validation
- Rate limiting

**Performance:**
- Response time optimization
- Caching strategies
- Concurrent request handling
- Resource management

**Monitoring:**
- Health checks
- Alerting on failures
- Performance dashboards
- Cost tracking

#### **Multi-Agent Systems**

**Why Multi-Agent:**
- Separation of concerns (specialized agents)
- Parallel processing
- Modular architecture
- Easier testing and maintenance

**Example Architecture:**
```
User → Orchestrator Agent
         ├─→ Search Agent (data discovery)
         ├─→ Query Agent (SQL generation)
         ├─→ Forecast Agent (predictions)
         └─→ Insights Agent (analysis)
```

#### **Agent2Agent (A2A) Protocol**

**What is A2A:**
- Open protocol for agent-to-agent communication
- Framework-agnostic (works across ADK, LangChain, CrewAI, etc.)
- Industry standard with 50+ partners (Microsoft, Salesforce, PayPal, etc.)

**A2A Capabilities:**

**1. Universal Interoperability**
- Agents from different vendors can collaborate
- No vendor lock-in
- Ecosystem growth

**2. Capability Discovery**
- **Agent Cards**: JSON documents describing agent capabilities
- Identity, skills, supported features
- Authentication requirements

**3. Secure Communication**
- HTTPS/TLS encryption
- JWT, OIDC authentication
- API key support
- Enterprise-grade security

**4. Stateless Interactions**
- A2A v0.2 supports stateless agent calls
- Easier scaling and reliability

**A2A Protocol Flow:**
```
1. Agent A discovers Agent B via Agent Card
2. Agent A authenticates with Agent B
3. Agent A sends task request to Agent B
4. Agent B processes and returns result
5. Agent A integrates result into workflow
```

**ADK + A2A:**
- Build A2A-compatible agents in ADK
- Consume remote A2A agents as tools
- Expose ADK agents via A2A protocol

**Quickstart: Consuming A2A Agent**
```python
from adk import Agent, A2ARemoteTool

# Connect to remote A2A agent
weather_agent = A2ARemoteTool(
    url="https://weather-agent.example.com/a2a"
)

# Use in your agent
my_agent = Agent(
    name="assistant",
    tools=[weather_agent]
)
```

**Example Multi-Agent Codelab:**
- Currency converter agent (A2A server)
- Travel planner agent (A2A client)
- Agents communicate via A2A protocol

---

## 🔧 Google Agent Development Kit (ADK) - Core Reference

### **What is ADK?**
Google's ADK is an open-source framework for building production-ready AI agents. It's the same framework that powers:
- Google Agentspace
- Google Customer Engagement Suite (CES)
- Enterprise Google agents

### **ADK Philosophy**
- **Model-agnostic**: Works with any LLM (Gemini, GPT, Claude, etc.)
- **Deployment-agnostic**: Run locally, cloud, edge
- **Framework-compatible**: Integrates with other systems
- **Production-ready**: Built for enterprise use

### **ADK Core Components**

**1. Agent**
```python
from adk import Agent

agent = Agent(
    name="my_agent",
    model="gemini-2.5-flash",
    tools=[search_tool, query_tool],
    system_prompt="You are a data analyst...",
    session_service=session_service
)
```

**2. Tools**
```python
from adk import tool

@tool
def search_data(query: str, database_id: str) -> dict:
    """Search for relevant columns in database"""
    # Tool implementation
    return results
```

**3. Sessions**
```python
from adk import InMemorySessionService

session_service = InMemorySessionService()
session = session_service.create_session()
```

**4. State Management**
```python
# In prompt
"Current database: {current_database}"

# In tool
def my_tool(arg: str, tool_context: ToolContext):
    db = tool_context.state.get("current_database")
```

**5. Memory**
```python
from adk import MemoryService

memory = MemoryService()
memory.add_session_to_memory(session)
results = memory.search_memory("past user preferences")
```

### **ADK Runtime & Agent Lifecycle**

**1. Agent Initialization**
- Load configuration
- Initialize model connection
- Register tools
- Setup session service

**2. Message Processing**
- Receive user input
- Load session state
- Generate model response
- Execute tool calls (if any)
- Update state
- Return response

**3. Tool Execution**
- Parse tool request
- Validate parameters
- Execute function
- Return result to agent
- Agent processes result

**4. Session Management**
- Create session on first message
- Load state for each turn
- Update state after each turn
- Close session when done

---

## 🎓 Capstone Project

### **Requirements**
Participants must design and implement an AI agent demonstrating **at least 3 capabilities** learned during the course.

### **Submission Components**
1. **Code**: Kaggle notebook OR GitHub repository
2. **Video**: 3-minute YouTube overview (public or unlisted)
3. **Documentation**: Written architecture and evaluation docs
4. **Writeup**: Kaggle competition submission (<1500 words)

### **Tracks (Choose One)**
1. **Enterprise Agents** - Business workflows, data analysis, automation
2. **Concierge Agents** - Customer support, recommendations, assistance
3. **Coding Agents** - Code generation, debugging, refactoring
4. **Freestyle** - Creative, experimental, innovative

### **Scoring (100 points max)**

| Category | Points | Details |
|----------|--------|---------|
| **The Pitch** | 30 | Problem definition + innovation |
| ├─ Core Concept | 15 | Relevance to track & originality |
| └─ Writeup | 15 | Clear communication |
| **Implementation** | 70 | Technical execution |
| ├─ Code Quality | 50 | 3+ concepts, working code |
| └─ Documentation | 20 | README, architecture, setup |
| **BONUS** | +20 | Extra credit |
| ├─ Gemini Use | 5 | Using Gemini models |
| ├─ Deployment | 5 | Deployed to cloud |
| └─ Video | 10 | Quality YouTube demo |

### **Winning Criteria**
- Top 3 per track (12 total winners)
- Prizes: Kaggle swag + social media recognition
- Judged on: Innovation, technical depth, clarity, production-readiness

### **Deadline**
**December 1, 2025, 11:59 AM PT** (extended from Nov 30)

---

## 🔗 Key Resources

### **Official Course Resources**
- **Course Page**: https://www.kaggle.com/learn-guide/5-day-agents
- **Competition**: https://www.kaggle.com/competitions/agents-intensive-capstone-project
- **Registration**: https://rsvp.withgoogle.com/events/google-ai-agents-intensive_2025

### **ADK Documentation**
- **Main Docs**: https://google.github.io/adk-docs/
- **Getting Started**: https://google.github.io/adk-docs/get-started/about/
- **Sessions**: https://google.github.io/adk-docs/sessions/
- **Tools**: https://google.github.io/adk-docs/tools-custom/
- **MCP**: https://google.github.io/adk-docs/mcp/
- **A2A**: https://google.github.io/adk-docs/a2a/
- **Evaluation**: https://google.github.io/adk-docs/evaluate/
- **Cloud Trace**: https://google.github.io/adk-docs/observability/cloud-trace/

### **GitHub Repositories with Course Materials**
1. **faith-ogun/5-day-ai-agents-intensive**
   - Most comprehensive resource
   - Daily codelabs and notebooks
   - Organized by day
   - Link: https://github.com/faith-ogun/5-day-ai-agents-intensive

2. **sdivyanshu90/5-Day-AI-Agents-Intensive-Course-with-Google**
   - Personal notes and solutions
   - Link: https://github.com/sdivyanshu90/5-Day-AI-Agents-Intensive-Course-with-Google

### **Google Codelabs**
- **Currency Agent (MCP, ADK, A2A)**: https://codelabs.developers.google.com/codelabs/currency-agent
- **Multi-Agent System**: https://codelabs.developers.google.com/codelabs/create-multi-agents-adk-a2a
- **Instavibe (Full Stack)**: https://codelabs.developers.google.com/instavibe-adk-multi-agents

### **Blog Posts & Tutorials**
- **ADK Sessions & Memory**: https://cloud.google.com/blog/topics/developers-practitioners/remember-this-agent-state-and-memory-with-adk
- **ADK + MCP**: https://cloud.google.com/blog/topics/developers-practitioners/use-google-adk-and-mcp-with-an-external-server
- **Multi-Agent with A2A**: https://medium.com/google-cloud/architecting-a-multi-agent-system-with-google-a2a-and-adk-4ced4502c86a

---

## 🎯 Key Takeaways for BQ Flow V2 Capstone

### **What BQ Flow Already Has (V1)**
✅ Production BigQuery AI integration
✅ 6 BigQuery AI features working
✅ Natural language to SQL
✅ Vector search for column discovery
✅ Time-series forecasting
✅ Structured insights generation
✅ Working UI (Chainlit)
✅ REST API (FastAPI)
✅ 10K+ lines of production code

### **What to Add for V2 (ADK Integration)**
🔨 **Tools** - Wrap existing BQ Flow functions as ADK tools
🔨 **Memory** - Add session state management with InMemorySessionService
🔨 **Evaluation** - Implement logging, metrics, tracing
🔨 **Documentation** - README_CAPSTONE.md with architecture
🔨 **Video** - 3-min YouTube demo
🔨 **Deployment** - Cloud Run deployment (bonus points)

### **Competitive Advantages**
1. **Real Infrastructure** - Not a toy demo, actual BigQuery AI
2. **Production Quality** - Enterprise-grade existing system
3. **Clear Evolution** - V1 → V2 story shows growth
4. **Measurable ROI** - Real business value calculation
5. **Technical Depth** - Multiple advanced concepts integrated

### **Timeline (Remaining)**
- **Days 1-5** (Nov 10-14): Course completed ✅
- **Day 8** (Nov 22): Build agent core + tools (2 hrs)
- **Day 9** (Nov 23): Add memory + evaluation (2 hrs)
- **Day 10** (Nov 24): Documentation + deployment (2 hrs)
- **Day 11** (Nov 28): Video + writeup (3 hrs)
- **Day 12** (Dec 1): Submit before deadline (1 hr)

**Total Remaining: 10 hours over next 11 days**

### **Success Probability**
- **Guaranteed**: Badge + certificate + portfolio piece
- **Likely (70%)**: Top 3 in Enterprise track
- **Stretch (30%)**: 1st place Enterprise track

---

## 📝 Implementation Notes

### **Agent Architecture for BQ Flow V2**
```
User Question
    ↓
ADK Agent (orchestration layer)
    ↓
[Session State: database, columns, results]
    ↓
Tools (ADK custom functions)
    ├─→ search_data (vector_search_columns)
    ├─→ query_data (generate_sql + execute)
    ├─→ forecast_data (generate_ai_forecast)
    └─→ generate_insights (structured_insights_v2)
    ↓
BigQuery AI (existing V1 engine)
    ├─→ ML.GENERATE_EMBEDDING
    ├─→ VECTOR_SEARCH
    ├─→ AI.GENERATE
    ├─→ AI.GENERATE_TABLE
    └─→ AI.FORECAST
    ↓
Results → Update State → Response
```

### **3 Required Concepts Mapped to BQ Flow**

**1. TOOLS** ✅
- Wrap 4 existing BQ Flow functions
- Use ADK `@tool` decorator
- Leverage existing production code
- Add proper error handling

**2. SESSIONS & MEMORY** ✅
- Use `InMemorySessionService`
- State: `{current_database, previous_queries, discovered_columns, last_results}`
- Enable multi-turn conversations
- Context retention across turns

**3. OBSERVABILITY** ✅
- Log all tool invocations (structured JSON)
- Track metrics: success rates, latency, costs
- Implement evaluation framework
- Simple dashboard (optional)

### **Code Structure**
```
src/agents/
├── __init__.py
├── bq_flow_agent.py      # Main ADK agent
├── tools.py               # Tool definitions (@tool decorated)
├── memory.py              # Session & state management
└── evaluation.py          # Metrics & logging
```

### **Integration Pattern**
```python
# Reuse existing BQ Flow code
from src.core.bigquery_ai import (
    vector_search_columns,
    generate_sql_with_context,
    execute_bigquery_query
)
from src.core.bigquery_ai_forecast import generate_ai_forecast
from src.core.bigquery_ai_generate_table import generate_structured_insights_v2

# Wrap as ADK tools
from adk import tool

@tool
def search_data(query: str, database_id: str, tool_context: ToolContext) -> dict:
    """Search for relevant columns using vector search"""
    config = get_config()
    results = vector_search_columns(
        query, database_id, config.client, config
    )
    # Update state
    tool_context.state["discovered_columns"] = results
    return results
```

---

## 🚀 Next Steps

1. ✅ **Complete Course Review** (this document)
2. 🔨 **Start Implementation** (Day 8: Nov 22)
   - Create `src/agents/` directory
   - Implement agent core with ADK
   - Wrap 4 tools
3. 🔨 **Add Memory & Eval** (Day 9: Nov 23)
   - Session state management
   - Metrics logging
4. 📝 **Document & Deploy** (Day 10: Nov 24)
   - README_CAPSTONE.md
   - Cloud Run deployment
5. 🎬 **Create Video** (Day 11: Nov 28)
   - 3-minute YouTube demo
6. 📤 **Submit** (Day 12: Dec 1)
   - Before 11:59 AM PT deadline

---

**Document Created:** November 20, 2025
**Status:** Course context compiled, ready for implementation
**Next Action:** Begin Day 8 implementation (agent core + tools)
