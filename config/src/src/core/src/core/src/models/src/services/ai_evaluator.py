"""AI evaluation service for applications."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from datetime import datetime
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import re

from src.core.config import settings
from src.models.application import Application

logger = logging.getLogger(__name__)

class AIEvaluator:
    """AI-powered application evaluator."""
    
    def __init__(self):
        """Initialize AI evaluator."""
        self.model = None
        self.scaler = StandardScaler()
        self.load_model()
        
        # Evaluation weights
        self.weights = {
            'academic': 0.4,
            'test_scores': 0.3,
            'experience': 0.15,
            'statement': 0.15
        }
    
    def load_model(self):
        """Load pre-trained model."""
        try:
            self.model = joblib.load(settings.AI_MODEL_PATH)
            logger.info("AI model loaded successfully")
        except FileNotFoundError:
            logger.warning("No pre-trained model found. Using rule-based evaluation.")
            self.model = None
    
    async def evaluate_application(self, application: Application) -> Dict[str, Any]:
        """Evaluate a single application."""
        try:
            # Extract features
            features = self.extract_features(application)
            
            # Calculate scores
            academic_score = self.evaluate_academic(application)
            test_score = self.evaluate_test_scores(application)
            experience_score = self.evaluate_experience(application)
            statement_score = self.evaluate_personal_statement(application)
            
            # Calculate overall score
            overall_score = (
                academic_score * self.weights['academic'] +
                test_score * self.weights['test_scores'] +
                experience_score * self.weights['experience'] +
                statement_score * self.weights['statement']
            )
            
            # Get AI recommendation
            recommendation = self.get_recommendation(overall_score, features)
            confidence = self.calculate_confidence(features)
            
            # Detailed analysis
            analysis = {
                'academic_analysis': self.analyze_academic(application),
                'test_analysis': self.analyze_test_scores(application),
                'experience_analysis': self.analyze_experience(application),
                'statement_analysis': self.analyze_statement(application),
                'strengths': self.identify_strengths(application),
                'weaknesses': self.identify_weaknesses(application),
                'risk_factors': self.identify_risks(application)
            }
            
            return {
                'overall_score': round(overall_score * 100, 2),
                'academic_score': round(academic_score * 100, 2),
                'test_score': round(test_score * 100, 2),
                'experience_score': round(experience_score * 100, 2),
                'statement_score': round(statement_score * 100, 2),
                'recommendation': recommendation,
                'confidence': round(confidence * 100, 2),
                'analysis': analysis,
                'evaluation_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating application: {str(e)}")
            return {
                'error': str(e),
                'overall_score': 0,
                'recommendation': 'error',
                'confidence': 0
            }
    
    async def batch_evaluate(self, applications: List[Application]) -> List[Dict]:
        """Evaluate multiple applications."""
        results = []
        for app in applications:
            result = await self.evaluate_application(app)
            results.append(result)
        return results
    
    def extract_features(self, application: Application) -> np.ndarray:
        """Extract numerical features for ML model."""
        features = []
        
        # GPA
        features.append(application.gpa if application.gpa else 0)
        
        # Test scores (normalized)
        test_scores = application.test_scores or {}
        sat_score = test_scores.get('sat', 0) / 1600 if test_scores.get('sat') else 0
        gre_score = test_scores.get('gre', 0) / 340 if test_scores.get('gre') else 0
        toefl_score = test_scores.get('toefl', 0) / 120 if test_scores.get('toefl') else 0
        
        features.extend([sat_score, gre_score, toefl_score])
        
        # Experience years
        work_exp = application.work_experience or []
        total_exp_years = sum(exp.get('years', 0) for exp in work_exp)
        features.append(min(total_exp_years / 10, 1.0))  # Normalize to max 10 years
        
        # Statement length and complexity
        statement = application.personal_statement or ""
        statement_length = len(statement.split()) / 1000  # Normalize to max 1000 words
        features.append(min(statement_length, 1.0))
        
        return np.array(features).reshape(1, -1)
    
    def evaluate_academic(self, application: Application) -> float:
        """Evaluate academic performance."""
        score = 0
        
        # GPA evaluation
        if application.gpa:
            if application.gpa >= 3.7:
                score += 1.0
            elif application.gpa >= 3.3:
                score += 0.8
            elif application.gpa >= 3.0:
                score += 0.6
            elif application.gpa >= 2.5:
                score += 0.4
            else:
                score += 0.2
        
        # Education quality (could be enhanced with university rankings)
        education = application.education_history or []
        if education:
            # Simple heuristic: more education = better
            score *= min(1 + (len(education) * 0.1), 1.3)
        
        return min(score, 1.0)
    
    def evaluate_test_scores(self, application: Application) -> float:
        """Evaluate standardized test scores."""
        scores = application.test_scores or {}
        score = 0
        count = 0
        
        # SAT
        if 'sat' in scores:
            sat = scores['sat']
            if sat >= 1500:
                score += 1.0
            elif sat >= 1400:
                score += 0.8
            elif sat >= 1300:
                score += 0.6
            elif sat >= 1200:
                score += 0.4
            else:
                score += 0.2
            count += 1
        
        # GRE
        if 'gre' in scores:
            gre = scores['gre']
            if gre >= 330:
                score += 1.0
            elif gre >= 320:
                score += 0.8
            elif gre >= 310:
                score += 0.6
            elif gre >= 300:
                score += 0.4
            else:
                score += 0.2
            count += 1
        
        # TOEFL/IELTS
        if 'toefl' in scores:
            toefl = scores['toefl']
            if toefl >= 110:
                score += 1.0
            elif toefl >= 100:
                score += 0.8
            elif toefl >= 90:
                score += 0.6
            elif toefl >= 80:
                score += 0.4
            else:
                score += 0.2
            count += 1
        
        return score / max(count, 1)
    
    def evaluate_experience(self, application: Application) -> float:
        """Evaluate work and research experience."""
        experience = application.work_experience or []
        if not experience:
            return 0.3  # Base score for no experience
        
        total_score = 0
        for exp in experience:
            years = exp.get('years', 0)
            role = exp.get('role', '').lower()
            
            # Score based on years
            years_score = min(years / 5, 1.0)  # Max score at 5 years
            
            # Score based on role relevance
            relevance_score = 0.5
            if any(keyword in role for keyword in ['research', 'intern', 'assistant']):
                relevance_score = 0.8
            if any(keyword in role for keyword in ['manager', 'lead', 'senior']):
                relevance_score = 1.0
            
            total_score += (years_score + relevance_score) / 2
        
        return min(total_score / len(experience), 1.0)
    
    def evaluate_personal_statement(self, application: Application) -> float:
        """Evaluate personal statement quality."""
        statement = application.personal_statement or ""
        if not statement:
            return 0.2
        
        score = 0
        
        # Length check
        words = len(statement.split())
        if words >= 500:
            score += 0.3
        elif words >= 300:
            score += 0.2
        else:
            score += 0.1
        
        # Keyword relevance (simple version)
        keywords = ['passion', 'interest', 'goal', 'learn', 'future', 'skill', 'experience']
        keyword_count = sum(1 for keyword in keywords if keyword in statement.lower())
        score += min(keyword_count / len(keywords), 0.3)
        
        # Structure check
        if '\n\n' in statement or statement.count('.') > 5:
            score += 0.2
        
        # Grammar/spelling (simplified)
        # In production, use a proper grammar checker
        common_errors = ['teh', 'recieve', 'seperate']
        error_count = sum(1 for error in common_errors if error in statement.lower())
        if error_count == 0:
            score += 0.2
        elif error_count <= 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_recommendation(self, score: float, features: np.ndarray) -> str:
        """Get AI recommendation based on score and ML model."""
        if self.model:
            # Use ML model for prediction
            try:
                prediction = self.model.predict_proba(features)[0]
                if prediction[1] > 0.8:
                    return 'strong_accept'
                elif prediction[1] > 0.6:
                    return 'accept'
                elif prediction[1] > 0.4:
                    return 'review'
                elif prediction[1] > 0.2:
                    return 'waitlist'
                else:
                    return 'reject'
            except:
                pass
        
        # Fallback to rule-based
        if score >= 0.85:
            return 'strong_accept'
        elif score >= 0.70:
            return 'accept'
        elif score >= 0.50:
            return 'review'
        elif score >= 0.30:
            return 'waitlist'
        else:
            return 'reject'
    
    def calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in the evaluation."""
        # Simple confidence based on data completeness
        completeness = np.mean(features > 0)
        return min(completeness * 1.2, 1.0)  # Boost but cap at 1.0
    
    def analyze_academic(self, application: Application) -> Dict:
        """Detailed academic analysis."""
        analysis = {
            'gpa_assessment': '',
            'transcript_quality': '',
            'academic_trend': ''
        }
        
        if application.gpa:
            if application.gpa >= 3.7:
                analysis['gpa_assessment'] = 'Excellent academic performance'
            elif application.gpa >= 3.3:
                analysis['gpa_assessment'] = 'Good academic performance'
            elif application.gpa >= 3.0:
                analysis['gpa_assessment'] = 'Satisfactory academic performance'
            else:
                analysis['gpa_assessment'] = 'Below average academic performance'
        
        return analysis
    
    def analyze_test_scores(self, application: Application) -> Dict:
        """Detailed test score analysis."""
        scores = application.test_scores or {}
        analysis = {}
        
        for test, score in scores.items():
            if test == 'sat':
                if score >= 1500:
                    analysis['sat'] = 'Excellent SAT performance'
                elif score >= 1400:
                    analysis['sat'] = 'Good SAT performance'
                else:
                    analysis['sat'] = 'Average SAT performance'
            elif test == 'gre':
                if score >= 330:
                    analysis['gre'] = 'Excellent GRE performance'
                elif score >= 320:
                    analysis['gre'] = 'Good GRE performance'
                else:
                    analysis['gre'] = 'Average GRE performance'
        
        return analysis
    
    def analyze_experience(self, application: Application) -> Dict:
        """Detailed experience analysis."""
        experience = application.work_experience or []
        analysis = {
            'total_years': sum(exp.get('years', 0) for exp in experience),
            'relevance': 'Relevant' if experience else 'No experience',
            'quality': 'High' if len(experience) > 2 else 'Moderate' if experience else 'Low'
        }
        return analysis
    
    def analyze_statement(self, application: Application) -> Dict:
        """Detailed statement analysis."""
        statement = application.personal_statement or ""
        analysis = {
            'length': len(statement.split()),
            'quality': 'Good' if len(statement.split()) > 300 else 'Needs improvement',
            'clarity': 'Clear' if statement and len(statement) > 500 else 'Could be clearer'
        }
        return analysis
    
    def identify_strengths(self, application: Application) -> List[str]:
        """Identify applicant strengths."""
        strengths = []
        
        if application.gpa and application.gpa >= 3.5:
            strengths.append('Strong academic record')
        
        scores = application.test_scores or {}
        if scores.get('sat', 0) >= 1400:
            strengths.append('Excellent SAT scores')
        if scores.get('gre', 0) >= 320:
            strengths.append('Excellent GRE scores')
        
        if application.work_experience and len(application.work_experience) >= 2:
            strengths.append('Relevant work experience')
        
        statement = application.personal_statement or ""
        if len(statement.split()) >= 500:
            strengths.append('Well-written personal statement')
        
        return strengths
    
    def identify_weaknesses(self, application: Application) -> List[str]:
        """Identify applicant weaknesses."""
        weaknesses = []
        
        if application.gpa and application.gpa < 3.0:
            weaknesses.append('Below average GPA')
        
        scores = application.test_scores or {}
        if scores.get('sat', 1600) < 1200:
            weaknesses.append('Low SAT scores')
        if scores.get('gre', 340) < 300:
            weaknesses.append('Low GRE scores')
        
        if not application.work_experience:
            weaknesses.append('No work experience')
        
        statement = application.personal_statement or ""
        if len(statement.split()) < 300:
            weaknesses.append('Personal statement too short')
        
        return weaknesses
    
    def identify_risks(self, application: Application) -> List[str]:
        """Identify potential risks."""
        risks = []
        
        # Academic risk
        if application.gpa and application.gpa < 2.5:
            risks.append('High academic risk')
        
        # Test score risk
        scores = application.test_scores or {}
        if scores.get('toefl', 120) < 80:
            risks.append('Language proficiency concern')
        
        # Experience risk
        if not application.work_experience and not application.education_history:
            risks.append('Limited background information')
        
        # Statement risk
        statement = application.personal_statement or ""
        if not statement:
            risks.append('Missing personal statement')
        
        return risks

# Create singleton instance
ai_evaluator = AIEvaluator()
