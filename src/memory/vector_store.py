"""
FAISS-based vector memory system for persistent pattern storage.
Implements advanced memory for coding patterns and user preferences.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, SecretStr


class CodePattern(BaseModel):
    """Represents a learned coding pattern."""
    pattern_id: str
    pattern_type: str  # "function", "class", "style", "workflow", etc.
    language: str
    description: str
    code_snippet: str
    metadata: Dict[str, Any]
    usage_count: int = 0
    last_used: str
    embedding: Optional[List[float]] = None


class VectorMemoryStore:
    """
    FAISS-based vector store for persistent coding pattern memory.
    Works like an advanced memory system to learn and recall patterns.
    """
    
    def __init__(self, persist_directory: str = "./memory_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize embeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07",
            google_api_key=SecretStr(api_key)
        )
        
        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.patterns: List[CodePattern] = []
        self.dimension = 768  # Google embedding dimension
        
        # Metadata storage
        self.metadata_file = self.persist_directory / "patterns_metadata.json"
        self.index_file = self.persist_directory / "faiss_index.pkl"
        
        # Load existing data
        self._load_memory()
    
    def _load_memory(self):
        """Load existing patterns and FAISS index."""
        try:
            # Load patterns metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    patterns_data = json.load(f)
                    self.patterns = [CodePattern(**p) for p in patterns_data]
                print(f"📚 Loaded {len(self.patterns)} coding patterns from memory")
            
            # Load FAISS index
            if self.index_file.exists() and self.patterns:
                with open(self.index_file, 'rb') as f:
                    self.index = pickle.load(f)
                if self.index is not None:
                    print(f"🔍 Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                self._create_new_index()
                
        except Exception as e:
            print(f"⚠️ Error loading memory: {e}. Starting fresh.")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for similarity
        print("🆕 Created new FAISS index")
    
    def _save_memory(self):
        """Persist patterns and index to disk."""
        try:
            # Save patterns metadata
            patterns_data = [p.dict() for p in self.patterns]
            with open(self.metadata_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save FAISS index
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.index, f)
                
            print(f"💾 Saved {len(self.patterns)} patterns to memory")
        except Exception as e:
            print(f"❌ Error saving memory: {e}")
    
    async def add_pattern(
        self, 
        pattern_type: str,
        language: str,
        description: str,
        code_snippet: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new coding pattern to memory.
        Returns the pattern ID.
        """
        try:
            # Generate embedding
            embedding = await self.embeddings.aembed_query(
                f"{description} {code_snippet}"
            )
            
            # Create pattern
            from datetime import datetime
            pattern_id = f"{pattern_type}_{language}_{len(self.patterns)}"
            pattern = CodePattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                language=language,
                description=description,
                code_snippet=code_snippet,
                metadata=metadata or {},
                last_used=datetime.now().isoformat(),
                embedding=embedding
            )
            
            # Add to memory
            self.patterns.append(pattern)
            
            # Add to FAISS index
            if self.index is not None:
                embedding_array = np.array([embedding], dtype=np.float32)
                self.index.add(embedding_array)
            
            # Save to disk
            self._save_memory()
            
            print(f"🧠 Learned new {pattern_type} pattern: {description[:50]}...")
            return pattern_id
            
        except Exception as e:
            print(f"❌ Error adding pattern: {e}")
            return ""
    
    async def find_similar_patterns(
        self, 
        query: str,
        top_k: int = 5,
        pattern_type: Optional[str] = None,
        language: Optional[str] = None
    ) -> List[Tuple[CodePattern, float]]:
        """
        Find similar coding patterns based on query.
        Returns patterns with similarity scores.
        """
        if not self.patterns or self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search FAISS index
            k = min(top_k * 2, len(self.patterns))
            scores, indices = self.index.search(query_vector, k)
            
            # Filter and format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.patterns):
                    pattern = self.patterns[idx]
                    
                    # Apply filters
                    if pattern_type and pattern.pattern_type != pattern_type:
                        continue
                    if language and pattern.language != language:
                        continue
                    
                    results.append((pattern, float(score)))
                    
                    if len(results) >= top_k:
                        break
            
            # Update usage counts
            from datetime import datetime
            for pattern, _ in results:
                pattern.usage_count += 1
                pattern.last_used = datetime.now().isoformat()
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching patterns: {e}")
            return []
    
    async def analyze_code_style(self, code: str, language: str) -> Dict[str, Any]:
        """
        Analyze code to extract style patterns using advanced techniques.
        """
        try:
            # Find similar patterns
            similar_patterns = await self.find_similar_patterns(
                f"analyze code style {code[:200]}",
                top_k=3,
                language=language
            )
            
            # Extract style insights
            style_analysis = {
                "language": language,
                "similar_patterns_found": len(similar_patterns),
                "style_preferences": {},
                "suggestions": []
            }
            
            if similar_patterns:
                # Analyze common patterns
                common_types = {}
                for pattern, score in similar_patterns:
                    pattern_type = pattern.pattern_type
                    if pattern_type in common_types:
                        common_types[pattern_type] += score
                    else:
                        common_types[pattern_type] = score
                
                style_analysis["common_patterns"] = common_types
                style_analysis["suggestions"] = [
                    f"Consider using {pattern.description}" 
                    for pattern, _ in similar_patterns[:2]
                ]
            
            return style_analysis
            
        except Exception as e:
            print(f"❌ Error analyzing code style: {e}")
            return {"error": str(e)}
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about stored patterns."""
        if not self.patterns:
            return {"total_patterns": 0}
        
        stats = {
            "total_patterns": len(self.patterns),
            "by_language": {},
            "by_type": {},
            "most_used": [],
            "recent_patterns": []
        }
        
        # Count by language and type
        for pattern in self.patterns:
            # By language
            if pattern.language in stats["by_language"]:
                stats["by_language"][pattern.language] += 1
            else:
                stats["by_language"][pattern.language] = 1
            
            # By type
            if pattern.pattern_type in stats["by_type"]:
                stats["by_type"][pattern.pattern_type] += 1
            else:
                stats["by_type"][pattern.pattern_type] = 1
        
        # Most used patterns
        sorted_patterns = sorted(self.patterns, key=lambda p: p.usage_count, reverse=True)
        stats["most_used"] = [
            {"id": p.pattern_id, "usage_count": p.usage_count, "description": p.description}
            for p in sorted_patterns[:5]
        ]
        
        # Recent patterns
        recent_patterns = sorted(self.patterns, key=lambda p: p.last_used, reverse=True)
        stats["recent_patterns"] = [
            {"id": p.pattern_id, "last_used": p.last_used, "description": p.description}
            for p in recent_patterns[:5]
        ]
        
        return stats
    
    async def learn_from_interaction(
        self, 
        user_query: str,
        code_generated: str,
        language: str,
        user_feedback: Optional[str] = None
    ):
        """
        Learn from user interactions using advanced pattern recognition.
        """
        try:
            # Determine pattern type based on query
            pattern_type = "general"
            if "function" in user_query.lower():
                pattern_type = "function"
            elif "class" in user_query.lower():
                pattern_type = "class"
            elif "test" in user_query.lower():
                pattern_type = "test"
            elif "refactor" in user_query.lower():
                pattern_type = "refactor"
            
            # Create metadata
            from datetime import datetime
            metadata = {
                "user_query": user_query,
                "user_feedback": user_feedback,
                "interaction_context": "code_generation",
                "timestamp": datetime.now().isoformat()
            }
            
            # Add pattern
            pattern_id = await self.add_pattern(
                pattern_type=pattern_type,
                language=language,
                description=f"Pattern from: {user_query[:100]}",
                code_snippet=code_generated,
                metadata=metadata
            )
            
            return pattern_id
            
        except Exception as e:
            print(f"❌ Error learning from interaction: {e}")
            return None
    
    def clear_memory(self):
        """Clear all stored patterns (use with caution!)."""
        self.patterns = []
        self._create_new_index()
        
        # Remove files
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        if self.index_file.exists():
            self.index_file.unlink()
        
        print("🗑️ Memory cleared")
