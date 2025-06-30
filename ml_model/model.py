import os
import json
import spacy
import torch
import numpy as np
import pandas as pd
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import PyPDF2
import fitz

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class JobRecommender:
    def __init__(self):
        # Initialize with CPU first to avoid immediate GPU memory allocation
        self.device = 'cpu'
        self.nlp = spacy.load("en_core_web_lg")
        self.skill_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher)
        
        # Load model after other components to manage GPU memory better
        self.initialize_model()
        
        # Load data
        self.df = pd.read_csv("camhr_cleaned_data.csv")
        self.df['job_text_lower'] = self.df['job_text'].fillna('').str.lower()
        
        # Load embeddings
        self.load_embeddings()

    def initialize_model(self):
        """Initialize model with proper device handling"""
        clear_gpu_memory()
        try:
            # Try GPU first
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.model = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    device='cuda',
                    cache_folder='./model_cache'
                )
            else:
                self.device = 'cpu'
                self.model = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    device='cpu',
                    cache_folder='./model_cache'
                )
        except RuntimeError as e:
            print(f"Failed to initialize on GPU: {e}. Falling back to CPU.")
            self.device = 'cpu'
            self.model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device='cpu',
                cache_folder='./model_cache'
            )

    def load_embeddings(self):
        """Load or compute embeddings with memory management"""
        clear_gpu_memory()
        embedding_file = "job_embeddings.npy"
        
        if os.path.exists(embedding_file):
            try:
                # Load numpy array first to control device placement
                embeddings_np = np.load(embedding_file)
                self.job_embeddings = torch.from_numpy(embeddings_np).to(self.device)
                return
            except Exception as e:
                print(f"Error loading embeddings: {e}. Recomputing...")
        
        # Compute embeddings in batches
        batch_size = 16 if self.device == 'cuda' else 64
        text_list = self.df['job_text'].tolist()
        embeddings = []
        
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings.cpu())  # Store on CPU
            clear_gpu_memory()
        
        # Combine and save
        self.job_embeddings = torch.cat(embeddings).to(self.device)
        np.save(embedding_file, self.job_embeddings.cpu().numpy())

    def extract_text_from_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            try:
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    if text.strip():
                        return text
                    else:
                        return self.extract_text_from_pdf_fitz(file_path)
            except Exception:
                return self.extract_text_from_pdf_fitz(file_path)
        elif ext == ".docx":
            return docx2txt.process(file_path)
        else:
            raise ValueError("Unsupported file type (.pdf/.docx only)")

    def extract_text_from_pdf_fitz(self, pdf_path):
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        return text

    def load_skills_from_json(self, file_path="skills.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data.get("skills", [])
        except Exception as e:
            print(f"Could not load custom skills from JSON: {e}")
            return []

    def extract_skills(self, text):
        nlp_skills = set()
        custom_skills_list = self.load_skills_from_json()
        
        try:
            annotations = self.skill_extractor.annotate(text)
            full_matches = annotations["results"].get("full_matches", [])
            ngram_matches = annotations["results"].get("ngram_scored", [])
            
            for match in full_matches + ngram_matches:
                if "doc_node_value" in match:
                    nlp_skills.add(match["doc_node_value"])
        except Exception as e:
            print(f"Error using SkillExtractor: {e}")
        
        keyword_skills = set(skill for skill in custom_skills_list 
                           if skill.lower() in text.lower())
        
        all_skills = nlp_skills.union(keyword_skills)
        normalized_skills = set(skill.strip().lower() for skill in all_skills)
        ignore = ["c", "d", "m", "a", "b", "e", "f", "g", "h", "i", "j", 
                 "k", "l", "n", "r", "o", "p", "q", "s", "t", "u", 
                 "v", "w", "x", "y", "z"]
        
        return [s for s in normalized_skills if s not in ignore]

    def recommend_jobs(self, cv_text, cv_skills):
        clear_gpu_memory()
        cv_skills_set = set(skill.lower() for skill in cv_skills)
        cv_skills_text = ' '.join(cv_skills_set)
        
        try:
            # Process CV embedding
            cv_embedding = self.model.encode(
                cv_skills_text,
                convert_to_tensor=True,
                show_progress_bar=False
            ).to(self.device)
            
            # Calculate similarity in chunks
            similarity_scores = []
            chunk_size = 512  # Smaller chunks for GPU memory
            
            for i in range(0, len(self.job_embeddings), chunk_size):
                chunk = self.job_embeddings[i:i + chunk_size]
                scores = util.cos_sim(cv_embedding, chunk)[0]
                similarity_scores.append(scores.cpu())
                del scores
                clear_gpu_memory()
                
            self.df['bert_match_score'] = torch.cat(similarity_scores).numpy()
            
            # TF-IDF processing (CPU-based)
            texts = self.df['job_text'].tolist() + [cv_skills_text]
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            
            cv_vector = tfidf_matrix[-1]
            job_vectors = tfidf_matrix[:-1]
            cosine_scores_tfidf = cosine_similarity(cv_vector, job_vectors).flatten()
            self.df['tfidf_match_score'] = cosine_scores_tfidf
            
            # Combined score
            self.df['final_score'] = (0.6 * self.df['bert_match_score'] + 
                                     0.4 * self.df['tfidf_match_score'])
            
            # Get matched skills
            self.df['matched_skills'] = self.df['job_text'].apply(
                lambda job_text: {skill for skill in cv_skills_set 
                                if skill in job_text.lower()}
            )
            
            # Sort and filter
            df_sorted = self.df.sort_values(by='final_score', ascending=False)
            top_matches = df_sorted[df_sorted['final_score'] > 0.3]
            top_matches = top_matches.drop_duplicates(
                subset=['Company Name', 'Job Title'], 
                keep='first'
            )
            
            return top_matches[['Job Title', 'Company Name', 'final_score', 
                              'bert_match_score', 'tfidf_match_score', 
                              'matched_skills', 'Link URL']].head(6).to_dict('records')
        
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("Switching to CPU due to GPU memory constraints")
                self.device = 'cpu'
                self.model = self.model.to('cpu')
                self.job_embeddings = self.job_embeddings.to('cpu')
                return self.recommend_jobs(cv_text, cv_skills)
            raise