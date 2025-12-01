Onya Knowledge Graph Demos. [Onya](https://github.com/OoriData/Onya) is a knowledge graph format and implementation that uses human-readable `.onya` files to represent structured knowledge. The name comes from Igbo "ọ́nyà" meaning web/network.

# OnyaKB Features

OgbujiPT's `OnyaKB` backend provides:

- **File-based knowledge graphs**: Load `.onya` files from a directory
- **In-memory storage**: No database required for static knowledge bases
- **Multiple search strategies**: Text search, type-based filtering, property matching
- **KBBackend protocol**: Compatible with OgbujiPT's unified KB system
- **Human-editable**: Edit `.onya` files directly and reload

# Demos

## 1. `simple_onya_demo.py`

Basic demonstration covering:
- Loading `.onya` files from a directory
- Text-based search across node properties
- Type-based filtering (e.g., find all Person nodes)
- Property-based search (e.g., find nodes with specific values)
- Individual node retrieval by IRI

Run:
```bash
python demo/kgraph/simple_onya_demo.py
```

# Onya File Format

Basic `.onya` format example:

```onya
# @docheader
* @document: http://example.org/mydata
* @base: http://example.org/entities/

# Alice [Person]
* name: Alice Smith
* age: 30
* bio: Software engineer who loves Python

# Bob [Person]
* name: Bob Jones
* occupation: Data Scientist
```

Key elements:
- **Document header**: Declares namespaces and base URI
- **Nodes**: Marked with `# NodeName [Type]`
- **Properties**: Listed with `* property: value`
- **Types**: Entities can have one or more types

# Creating Your Own Knowledge Graph

1. **Create `.onya` files** in a directory:
   ```bash
   mkdir my_knowledge
   ```

2. **Write your knowledge** in `.onya` format:
   ```bash
   cat > my_knowledge/people.onya << 'EOF'
   # @docheader
   * @document: http://example.org/mykg
   * @base: http://example.org/

   # Person1 [Person]
   * name: Your Name
   * role: Your Role
   * expertise: Your Expertise
   EOF
   ```

3. **Load and search**:
   ```python
   from ogbujipt.store.kgraph import OnyaKB

   kb = OnyaKB(folder_path='./my_knowledge')
   await kb.setup()

   # Search your knowledge
   async for result in kb.search('expertise', limit=5):
       print(result.content)
   ```

# Integration with Other OgbujiPT Features

## Hybrid Search with Vectors

Combine graph-based search with vector search:

```python
from ogbujipt.store.kgraph import OnyaKB
from ogbujipt.store.ram import RAMDataDB
from ogbujipt.retrieval import TypeSearch, DenseSearch, HybridSearch
from sentence_transformers import SentenceTransformer

# Load knowledge graph
kg = OnyaKB(folder_path='./knowledge')
await kg.setup()

# Create vector store
model = SentenceTransformer('all-MiniLM-L6-v2')
vector_db = RAMDataDB(embedding_model=model, collection_name='docs')
await vector_db.setup()

# Add graph content to vector store for semantic search
async for result in kg.search('', limit=0):  # Get all nodes
    await vector_db.insert(result.content, result.metadata)

# Hybrid search across both
hybrid = HybridSearch(
    strategies=[DenseSearch(), TypeSearch(type_iri='http://schema.org/Person')],
)

async for result in hybrid.execute('machine learning expert',
                                   backends=[kg, vector_db],
                                   limit=5):
    print(result.content, result.score)
```

## GraphRAG Applications

Use Onya KG as the knowledge layer in RAG applications:

```python
from ogbujipt.store.kgraph import OnyaKB
from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat

# Load domain knowledge
kb = OnyaKB(folder_path='./domain_knowledge')
await kb.setup()

# Retrieve relevant knowledge
contexts = []
async for result in kb.search(user_query, limit=3):
    contexts.append(result.content)

# Build RAG prompt
context_text = '\n\n'.join(contexts)
prompt = f"""Based on this knowledge:

{context_text}

Question: {user_query}"""

# Get LLM response
llm = openai_chat_api(base_url='http://localhost:8000')
response = await llm(prompt_to_chat(prompt))
print(response.first_choice_text)
```

# Use Cases

## Static Knowledge Bases
- **Ontologies**: Load domain ontologies (schema.org, FOAF, etc.)
- **Taxonomies**: Product catalogs, classification systems
- **Reference data**: Countries, currencies, standards
- **Company knowledge**: Org charts, procedures, policies

## Human-Curated Knowledge
- **Expert knowledge**: Subject matter expertise in structured form
- **Documentation**: Technical docs as knowledge graphs
- **Metadata**: Structured descriptions of assets/resources

## Embedded Applications
- **No database required**: Bundle knowledge with your application
- **Version controlled**: `.onya` files in git for change tracking
- **Reviewable**: Human-readable format for peer review
- **Composable**: Multiple `.onya` files for modular knowledge

# Architecture Notes

## Read-Only by Design

`OnyaKB` is intentionally read-only:
- **insert()** and **delete()** raise `NotImplementedError`
- Edit `.onya` files directly using your text editor
- Reload by calling `cleanup()` then `setup()` again
- This design encourages human curation and version control

## In-Memory Performance

All nodes are loaded into memory:
- **Fast**: No database queries, instant lookups
- **Scalable**: Suitable for graphs with up to ~100K nodes
- **Simple**: No external dependencies or setup
- **Predictable**: Performance independent of query complexity

## Search Strategies

Three built-in strategies:
1. **Text search** (`kb.search()`): Substring matching across properties
2. **Type search** (`TypeSearch`): Filter by entity type
3. **Property search** (`PropertySearch`): Match specific property values

For semantic search, combine with vector stores using hybrid strategies.

# Prerequisites

```bash
# Easiest to just use the "mega" package, with all demo requirements
uv pip install -U ".[mega]"
```

# References

- **Onya**: [https://github.com/OoriData/Onya](https://github.com/OoriData/Onya)
- **OgbujiPT Documentation**: [https://github.com/OoriData/OgbujiPT](https://github.com/OoriData/OgbujiPT)
- **Knowledge Graphs**: [https://en.wikipedia.org/wiki/Knowledge_graph](https://en.wikipedia.org/wiki/Knowledge_graph)
- **GraphRAG**: [https://arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130)

---

**Need help?** Open an issue at [OgbujiPT GitHub](https://github.com/OoriData/OgbujiPT/issues)
