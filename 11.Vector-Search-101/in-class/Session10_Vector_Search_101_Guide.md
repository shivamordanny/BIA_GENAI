# Session 10: Vector Search 101
## From Text to Numbers — How Machines Find Similar Content

**Module:** 4 - Retrieval-Augmented Generation & Multimodal Systems  
**Level:** Beginner  
**Duration:** 3 hours  
**Target Audience:** BIA (Boston Institute of Analytics), Pune  
**Key Tools:** Google Gemini Embeddings, OpenAI Embeddings, FAISS  

---

## 1. Quick Recap: From Session 9 to Session 10

| Aspect | Session 9: Agent Tools | Session 10: Vector Search |
|--------|----------------------|-------------------------|
| **Focus** | Agents interact with APIs/tools | Agents need to *find* knowledge |
| **Problem** | How do agents take actions? | How do agents search knowledge bases? |
| **Solution** | Function calling | Vector embeddings + similarity search |
| **Next Stop** | Session 11: RAG | Putting embeddings into search pipelines |

**Why This Matters:**  
Agents need knowledge to make smart decisions. In Session 11 (RAG), we'll use embeddings to let agents retrieve relevant documents before answering questions. Today, we build the foundation: *understanding how text becomes searchable numbers*.

---

## 2. What Are Embeddings?

### The Core Idea

An **embedding** is a vector (list of numbers) that represents the *meaning* of text.

**The Problem:**  
Computers can't directly compare words like "king" and "queen" or decide which is more similar to "crown". Text is symbolic; machines need numbers.

**The Solution:**  
Convert text → numbers → compare numbers.

### Why Vectors?

Vectors let us use **math** to measure similarity. If two texts have similar meanings, their vectors should be *close* in space.

### Visual Analogy

Imagine a word space where:
- **"King"** is at position `[0.5, 0.8, 0.1]`
- **"Queen"** is at position `[0.6, 0.75, 0.15]` (similar direction)
- **"Banana"** is at position `[0.1, 0.2, 0.9]` (far away)

The king and queen vectors point in similar directions. The angle between them is small. The angle between king and banana is large. This is **cosine similarity**.

### Dimensions

Each number in the vector captures one aspect of *semantic meaning*:
- Position 1 might capture "royalty"
- Position 2 might capture "gender presentation"
- Position 3 might capture "fruit-ness"

Real embeddings have hundreds of dimensions. Dimension 47 might capture something we can't label. That's okay—the model learns it.

### Simple Python Example

```python
# This is what an embedding looks like
text = "The cat sat on the mat"
embedding = [0.1, -0.5, 0.3, 0.8, -0.2, ...]  # 768 numbers for Gemini
print(f"Text: {text}")
print(f"Embedding dimension: {len(embedding)}")  # e.g., 768
print(f"First 5 values: {embedding[:5]}")
```

---

## 3. Creating Embeddings with LangChain

LangChain provides a unified interface for multiple embedding models.

### Setup

```python
# Install required libraries
# pip install langchain-google-genai langchain-openai langchain-community faiss-cpu numpy

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

# Set your API keys
os.environ["GOOGLE_API_KEY"] = "your-key-here"
os.environ["OPENAI_API_KEY"] = "your-key-here"
```

### Google Gemini Embeddings (Primary)

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Embed a single query
query = "What is machine learning?"
query_embedding = embeddings.embed_query(query)
print(f"Query embedding length: {len(query_embedding)}")  # 768
print(f"First 5 values: {query_embedding[:5]}")
```

### OpenAI Embeddings (Alternative)

```python
from langchain_openai import OpenAIEmbeddings

# Initialize OpenAI embeddings
embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed the same query
query_embedding_openai = embeddings_openai.embed_query(query)
print(f"OpenAI embedding length: {len(query_embedding_openai)}")  # 1536
```

### Embedding Multiple Documents

```python
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand text.",
    "Computer vision enables machines to interpret images.",
]

# Embed all documents at once
doc_embeddings = embeddings.embed_documents(documents)
print(f"Number of document embeddings: {len(doc_embeddings)}")  # 4
print(f"Each embedding has {len(doc_embeddings[0])} dimensions")  # 768
```

### What the Output Looks Like

```python
# An embedding is a list of floats
example_embedding = [0.0123, -0.0456, 0.0789, ..., 0.0234]

# Properties:
# - Typically 768-3072 dimensions
# - Float values roughly in range [-1, 1]
# - Direction matters more than magnitude (for text)
```

---

## 4. Similarity Metrics

Now that we have embeddings, how do we compare them? Three main approaches:

### 4.1 Cosine Similarity

**What it measures:** The angle between two vectors (0° to 180°).

**Formula:**
```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```
Where `·` is dot product and `||A||` is magnitude.

**Range:** 0 to 1 (for normalized vectors)
- 1 = identical direction (identical meaning)
- 0 = perpendicular (completely different)

**Why it's best for text:** Cosine similarity ignores magnitude and focuses on direction. Two documents can use different vocabulary lengths but still be about the same topic.

**Python Implementation:**

```python
import numpy as np

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)

# Test
emb1 = [1, 0, 1]
emb2 = [1, 0, 1]
emb3 = [0, 1, 0]

print(f"Similarity (same): {cosine_similarity(emb1, emb2)}")  # 1.0
print(f"Similarity (different): {cosine_similarity(emb1, emb3)}")  # 0.0
```

### 4.2 Euclidean Distance

**What it measures:** Straight-line distance between two points.

**Formula:**
```
euclidean_distance(A, B) = sqrt((a1-b1)² + (a2-b2)² + ... + (an-bn)²)
```

**Range:** 0 to ∞
- 0 = identical vectors
- Larger = more different

**Note:** For normalized embeddings (magnitude = 1), Euclidean distance and cosine similarity give the same ranking. Use Euclidean for convenience if vectors are already normalized.

**Python Implementation:**

```python
def euclidean_distance(vec_a, vec_b):
    """Compute Euclidean distance between two vectors."""
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    return np.linalg.norm(vec_a - vec_b)

# Test
print(f"Distance (same): {euclidean_distance(emb1, emb2)}")  # 0.0
print(f"Distance (different): {euclidean_distance(emb1, emb3)}")  # ~1.41
```

### 4.3 Dot Product

**What it measures:** A scaled version of cosine similarity.

**Formula:**
```
dot_product(A, B) = a1*b1 + a2*b2 + ... + an*bn
```

**When to use:** If vectors are already normalized (magnitude = 1), dot product ≈ cosine similarity. Slightly faster to compute. FAISS uses this.

**Python Implementation:**

```python
def dot_product(vec_a, vec_b):
    """Compute dot product between two vectors."""
    return np.dot(np.array(vec_a), np.array(vec_b))
```

### Comparison Table

| Metric | Best For | Range | How to Interpret |
|--------|----------|-------|------------------|
| **Cosine Similarity** | Text embeddings | 0 to 1 | Higher = more similar |
| **Euclidean Distance** | Normalized vectors | 0 to ∞ | Lower = more similar |
| **Dot Product** | Fast normalized search | -∞ to ∞ | Higher = more similar |

**Recommendation:** For text embeddings, use **cosine similarity**. It's intuitive and treats all vectors fairly regardless of magnitude.

### Hands-On: Compare Sentences

```python
sentences = [
    "The cat sat on the mat",
    "A cat is sitting on a mat",
    "Dogs love to play fetch",
]

# Embed the sentences
embeddings_list = embeddings.embed_documents(sentences)

# Compute pairwise similarities
print("Cosine Similarity Matrix:")
for i, sent_i in enumerate(sentences):
    for j, sent_j in enumerate(sentences):
        sim = cosine_similarity(embeddings_list[i], embeddings_list[j])
        print(f"{sent_i[:30]:32} <-> {sent_j[:30]:32}: {sim:.4f}")

# Output:
# The cat sat on the mat       <-> The cat sat on the mat       : 1.0000
# The cat sat on the mat       <-> A cat is sitting on a mat    : 0.9543
# The cat sat on the mat       <-> Dogs love to play fetch      : 0.5234
```

---

## 5. Embedding Model Cost Comparison (2026)

Choosing an embedding model means balancing **cost**, **quality**, **dimensions**, and **speed**.

| Model | Provider | Price/1M Tokens | Dimensions | Best For | Free Tier? |
|-------|----------|-----------------|-----------|----------|-----------|
| **text-embedding-3-small** | OpenAI | $0.02 | 1536 | Budget-conscious, good quality | No |
| **text-embedding-3-large** | OpenAI | $0.13 | 3072 | High quality, fewer documents needed | No |
| **gemini-embedding-001** | Google | $0.15 | 768 | Good quality, English-focused | Yes* |
| **gemini-embedding-2-preview** | Google | $0.20 | 1024 | **Multimodal** (text + images) | Yes* |
| **Mistral Embed** | Mistral | $0.01 | 1024 | **Cheapest**, still decent quality | No |

**Key Insights:**

1. **Cheapest ≠ Best:** Mistral Embed is $0.01/1M but not as battle-tested as OpenAI.
2. **Quality vs. Cost:** OpenAI text-embedding-3-small offers the best value: $0.02 and tops many benchmarks.
3. **Multimodal Future:** Google's embedding-2-preview lets you embed text AND images together (for Session 12).
4. **Gemini for Learning:** This course uses Gemini because it has a generous free tier for development.

### Free Tier Notes

- **Google Gemini:** ~$300/month free credit includes embeddings API
- **OpenAI:** No free embeddings; pay-as-you-go ($0.02+)
- **Mistral:** Some free tier available, check current docs

### Cost Estimation

If you embed 1M documents (typical small knowledge base):

| Model | Total Cost |
|-------|-----------|
| Mistral | $0.01 |
| OpenAI small | $0.02 |
| OpenAI large | $0.13 |
| Gemini | $0.15 |

For this course: **Use Google Gemini** (primary) or **OpenAI small** (comparison).

---

## 6. Introduction to FAISS

### What is FAISS?

**FAISS** = "Facebook AI Similarity Search" (now Meta)  
An open-source library for finding similar items in large vector databases *very fast*.

### Why FAISS?

1. **Free** – Open source, no licensing costs
2. **Local** – Runs on your machine, no server needed
3. **Fast** – Millions of vectors in milliseconds
4. **Integrates with LangChain** – Easy API

### Core Concept: The Index

A **FAISS index** is a searchable data structure for vectors. Think of it like a library card catalog:

- Without index: search every book one-by-one (slow)
- With index: look up the catalog, find the right shelf (fast)

### Index Types for Beginners

#### IndexFlatL2

- **What it does:** Brute-force search using Euclidean distance
- **Speed:** O(n) — checks all vectors
- **Accuracy:** 100% — exact results
- **Best for:** Learning, small datasets (<10k vectors)

```python
import faiss
import numpy as np

# Create a simple index
index = faiss.IndexFlatL2(768)  # 768-dimensional vectors

# Add vectors
vectors = np.random.random((100, 768)).astype('float32')
index.add(vectors)

# Search
query_vector = np.random.random((1, 768)).astype('float32')
distances, indices = index.search(query_vector, k=5)  # Top 5
print(f"Nearest 5 vectors: {indices[0]}")
```

#### IndexFlatIP

- **What it does:** Inner product search (dot product)
- **Best for:** Normalized embeddings, cosine similarity
- **Speed:** Fast, exact

```python
# For normalized embeddings
index_ip = faiss.IndexFlatIP(768)
index_ip.add(vectors)
```

### Simple FAISS Workflow

1. **Create index** – Specify dimension and distance metric
2. **Add vectors** – Load embeddings into the index
3. **Search** – Query with a vector, get k nearest neighbors
4. **Return results** – Get indices and distances

---

## 7. Building a FAISS Index from Scratch

### Pure FAISS Approach (Understand What's Happening)

This approach shows you exactly how FAISS works:

```python
import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Step 1: Create sample documents
documents = [
    "Python is a popular programming language.",
    "Java is used for enterprise applications.",
    "JavaScript powers web browsers.",
    "C++ is known for high performance.",
    "SQL is used for database queries.",
    "Machine learning models process data.",
    "Neural networks have multiple layers.",
    "Deep learning powers image recognition.",
]

# Step 2: Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Step 3: Embed all documents
print("Embedding documents...")
doc_embeddings = embeddings.embed_documents(documents)
doc_vectors = np.array(doc_embeddings).astype('float32')
print(f"Shape: {doc_vectors.shape}")  # (8, 768)

# Step 4: Create and populate index
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_vectors)
print(f"Index contains {index.ntotal} vectors")

# Step 5: Create a query and search
query_text = "programming languages"
query_embedding = embeddings.embed_query(query_text)
query_vector = np.array([query_embedding]).astype('float32')

k = 3  # Top 3 results
distances, indices = index.search(query_vector, k)

# Step 6: Display results
print(f"\nQuery: {query_text}")
print("Top results:")
for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
    print(f"{i+1}. [{distance:.4f}] {documents[idx]}")
```

**Output:**
```
Index contains 8 vectors

Query: programming languages
Top results:
1. [0.0234] Python is a popular programming language.
2. [0.5643] Java is used for enterprise applications.
3. [0.6234] JavaScript powers web browsers.
```

### LangChain FAISS Integration (Simpler API)

LangChain wraps FAISS to make it even easier:

```python
from langchain_community.vectorstores import FAISS

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Create FAISS vector store directly from documents
documents = [
    "Python is a popular programming language.",
    "Java is used for enterprise applications.",
    "JavaScript powers web browsers.",
    "C++ is known for high performance.",
    "SQL is used for database queries.",
    "Machine learning models process data.",
    "Neural networks have multiple layers.",
    "Deep learning powers image recognition.",
]

# Create vector store
vector_store = FAISS.from_texts(documents, embeddings)

# Search
query = "programming languages"
results = vector_store.similarity_search(query, k=3)

print(f"Query: {query}")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content}")
```

### Saving and Loading an Index

```python
# Save the vector store
vector_store.save_local("my_faiss_index")

# Load it later
from langchain_community.vectorstores import FAISS
loaded_vector_store = FAISS.load_local("my_faiss_index", embeddings)

# Use it
results = loaded_vector_store.similarity_search("programming", k=2)
```

---

## 8. Putting It Together: Mini Search Engine

Let's build a simple document search engine that mirrors how RAG systems work.

```python
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

class SimpleSearchEngine:
    def __init__(self, documents, embedding_model="models/gemini-embedding-001"):
        """Initialize search engine with documents."""
        self.documents = documents
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_texts(documents, self.embeddings)
    
    def search(self, query, k=3):
        """Search for top k similar documents."""
        results = self.vector_store.similarity_search_with_scores(query, k=k)
        return results
    
    def display_results(self, query, k=3):
        """Pretty print search results."""
        results = self.search(query, k=k)
        
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        for i, (doc, score) in enumerate(results, 1):
            # Convert distance to similarity (lower distance = higher similarity)
            similarity = 1 / (1 + score)  # Rough conversion
            print(f"\n{i}. Similarity: {similarity:.4f} (Distance: {score:.4f})")
            print(f"   {doc.page_content}")
        print()

# Create sample FAQ database
faq_documents = [
    "What is Python? Python is a high-level, interpreted programming language known for simplicity and readability.",
    "How do I install Python? Download from python.org and run the installer for your operating system.",
    "What is machine learning? Machine learning is a subset of AI that enables systems to learn from data.",
    "What is a neural network? A neural network is a computational model inspired by biological neurons.",
    "How do embeddings work? Embeddings convert text into numerical vectors that capture semantic meaning.",
    "What is FAISS? FAISS is a library for fast similarity search in high-dimensional vector spaces.",
    "How do I create embeddings? Use pre-trained models like Google Gemini or OpenAI to convert text to vectors.",
    "What is cosine similarity? Cosine similarity measures the angle between two vectors (0 to 1).",
    "What is a vector database? A vector database stores embeddings and supports fast similarity search.",
    "What is RAG? RAG (Retrieval-Augmented Generation) combines retrieval and generation for better answers.",
]

# Initialize search engine
search_engine = SimpleSearchEngine(faq_documents)

# Test queries
test_queries = [
    "How do I use Python?",
    "What is machine learning?",
    "Tell me about vector search",
]

for query in test_queries:
    search_engine.display_results(query, k=3)
```

**Output Example:**
```
======================================================================
Query: How do I use Python?
======================================================================

1. Similarity: 0.6832 (Distance: 0.4641)
   What is Python? Python is a high-level, interpreted programming language known for simplicity and readability.

2. Similarity: 0.5421 (Distance: 0.8438)
   How do I install Python? Download from python.org and run the installer for your operating system.

3. Similarity: 0.4103 (Distance: 1.4521)
   How do embeddings work? Embeddings convert text into numerical vectors that capture semantic meaning.
```

---

## 9. Exercises

### Exercise 1: Build a FAQ Search System

Create a search engine for a small FAQ dataset:

```python
# TODO: Complete this exercise
faq = [
    "What is the capital of France? Paris is the capital of France.",
    "How many continents are there? There are seven continents.",
    "What is the largest ocean? The Pacific Ocean is the largest.",
    # Add 5 more FAQs...
]

# 1. Initialize embeddings
# 2. Create FAISS vector store from FAQs
# 3. Test with 3 different queries
# 4. Print top 2 results for each query with similarity scores
```

**Learning Goals:**
- Embed multiple documents
- Create and search a FAISS index
- Interpret similarity scores

---

### Exercise 2: Compare Cosine Similarity vs. Euclidean Distance

Compare how different metrics rank the same vectors:

```python
import numpy as np

sentences = [
    "The cat sat on the mat.",
    "A cat is sitting on a mat.",
    "The dog played in the park.",
    "Cats and dogs are pets.",
]

# 1. Embed all sentences
# 2. Compute cosine similarity between first sentence and others
# 3. Compute euclidean distance between first sentence and others
# 4. Create a table comparing the rankings
# 5. What do you notice?

# Expected insight: Rankings should be very similar for normalized embeddings
```

**Learning Goals:**
- Compute similarity metrics manually
- Understand how different metrics rank vectors
- See why cosine similarity is preferred

---

## 10. Quick Reference Card

### Key Imports

```python
# Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# FAISS (low-level)
import faiss
import numpy as np

# Linear algebra
from numpy import dot, linalg
```

### Creating Embeddings

```python
# Google Gemini (primary)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# OpenAI alternative
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Single text
vec = embeddings.embed_query("Your text here")

# Multiple texts
vecs = embeddings.embed_documents(["text1", "text2", "text3"])
```

### Similarity Metrics (NumPy)

```python
# Cosine similarity
def cosine_sim(a, b):
    return dot(a, b) / (linalg.norm(a) * linalg.norm(b))

# Euclidean distance
def euclidean_dist(a, b):
    return linalg.norm(np.array(a) - np.array(b))

# Dot product
def dot_prod(a, b):
    return dot(a, b)
```

### FAISS Operations

```python
# Create index
import faiss
index = faiss.IndexFlatL2(768)  # For Gemini embeddings

# Add vectors
vectors = np.array(embeddings).astype('float32')
index.add(vectors)

# Search
query_vec = np.array([query_embedding]).astype('float32')
distances, indices = index.search(query_vec, k=5)
```

### LangChain Vector Store

```python
# Create from texts
vs = FAISS.from_texts(["doc1", "doc2"], embeddings)

# Search
results = vs.similarity_search("query", k=3)

# Search with scores
results_scored = vs.similarity_search_with_scores("query", k=3)

# Save/load
vs.save_local("index_path")
vs = FAISS.load_local("index_path", embeddings)
```

### Common Patterns

```python
# Pattern 1: Simple search
vector_store = FAISS.from_texts(docs, embeddings)
top_3 = vector_store.similarity_search("query", k=3)

# Pattern 2: Batch embed
all_vectors = embeddings.embed_documents(documents)

# Pattern 3: Re-embed query
query_vec = embeddings.embed_query(user_query)
similarity = cosine_similarity(query_vec, doc_vec)
```

---

## Teaching Tips

### For Students

1. **Start with intuition:** Vectors are just lists of numbers. Similarity is just measuring distance.
2. **Play with numbers:** Run the embedding code and print outputs. See the actual vectors.
3. **Experiment with queries:** Try variations: "programming language" vs. "coding language" vs. "software tool".
4. **Save/load practice:** Get comfortable with `save_local()` and `load_local()` now—you'll need it in Session 11 for RAG.

### Common Mistakes to Avoid

1. **Forgetting to normalize:** Some similarity metrics require normalized vectors. Check your embedding model's docs.
2. **Wrong dimension:** If your embeddings are 768-dimensional, your FAISS index must be `IndexFlatL2(768)`, not 1536.
3. **Confusing distance vs. similarity:** Lower Euclidean distance = more similar. Higher cosine similarity = more similar. Keep a reference card.
4. **Using wrong metric for your data:** Cosine similarity for text embeddings. Euclidean distance for other vectors.

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "API key not found" | Set `GOOGLE_API_KEY` or `OPENAI_API_KEY` in environment |
| "faiss module not found" | Run `pip install faiss-cpu` |
| "Dimension mismatch" | Ensure embedding dimension matches index dimension |
| "Search returns wrong results" | Check if documents are actually embedded; verify similarity metric |

---

## Looking Ahead to Session 11

Session 10 teaches the *pieces* of a RAG system:
- Embeddings (Session 10 ✓)
- Vector search (Session 10 ✓)
- Metrics (Session 10 ✓)

Session 11 will *assemble* these pieces:
- Load documents
- Split into chunks
- Embed all chunks
- Create FAISS index
- **Use index to augment LLM prompts**

Everything you learn today is a building block for next session's RAG pipeline.

---

## Resources

### Official Documentation

- **Google Generative AI (Gemini):** https://ai.google.dev/docs
- **LangChain FAISS:** https://python.langchain.com/docs/integrations/vectorstores/faiss
- **FAISS GitHub:** https://github.com/facebookresearch/faiss
- **OpenAI Embeddings:** https://platform.openai.com/docs/guides/embeddings

### Recommended Readings

- "Embeddings Explained" — What are word embeddings and how do they work?
- "MTEB Leaderboard" — Compare embedding models: https://huggingface.co/spaces/mteb/leaderboard
- "An Introduction to Vector Databases" — Understand the landscape

### Key Papers (Optional Deeper Dive)

- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Learning to Compare: Relation Network for Few-Shot Learning

---

## Appendix: Full Code Examples

### Complete Mini Search Engine (Copy-Paste Ready)

```python
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Set API key
os.environ["GOOGLE_API_KEY"] = "your-key-here"

# Sample documents
documents = [
    "Python is a versatile programming language suitable for beginners and experts.",
    "JavaScript runs in web browsers and enables interactive web experiences.",
    "Java is widely used in enterprise applications and large-scale systems.",
    "C++ offers high performance and is used in systems programming.",
    "Ruby is known for productivity and has a focus on developer happiness.",
    "Go is designed for concurrent programming and cloud applications.",
    "Rust provides memory safety without garbage collection.",
    "TypeScript adds static typing to JavaScript for better code quality.",
]

# Initialize embeddings
print("Initializing embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Create vector store
print("Creating FAISS index...")
vector_store = FAISS.from_texts(documents, embeddings)

# Test queries
queries = [
    "What language is best for web development?",
    "Which language is fastest and most efficient?",
    "What language should a beginner learn?",
]

print("\nSearching...\n")
for query in queries:
    print(f"Query: {query}")
    results = vector_store.similarity_search_with_scores(query, k=2)
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. (distance={score:.4f}) {doc.page_content}")
    print()
```

### Complete Pure FAISS Example

```python
import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings

documents = [
    "Machine learning powers recommendation systems.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand text.",
    "Computer vision enables image recognition and analysis.",
]

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Embed documents
doc_embeddings = embeddings_model.embed_documents(documents)
doc_vectors = np.array(doc_embeddings).astype('float32')

# Create FAISS index
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_vectors)

# Query
query = "neural networks and deep learning"
query_emb = embeddings_model.embed_query(query)
query_vec = np.array([query_emb]).astype('float32')

distances, indices = index.search(query_vec, k=2)

print(f"Query: {query}")
for idx, distance in zip(indices[0], distances[0]):
    print(f"  [{distance:.4f}] {documents[idx]}")
```

---

## Summary

**Session 10 taught you:**
1. What embeddings are (text → numbers)
2. How to create embeddings (LangChain + Gemini/OpenAI)
3. How to measure similarity (cosine, Euclidean, dot product)
4. How to build a vector search index (FAISS)
5. How to search for similar documents

**You now understand the foundation of RAG systems.** Next session, you'll use these tools to augment an LLM with retrieved knowledge.

**Keep practicing:** Small changes in queries return different results. Experiment with similarity metrics. Try different embedding models. Build intuition.

---

**Session 10 Complete**  
*Next: Session 11 - Retrieval-Augmented Generation (RAG)*
