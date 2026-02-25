"""Application database models."""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator

from src.core.database import Base

# SQLAlchemy Models
class Application(Base):
    """Application model."""
    
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String(50), unique=True, index=True, nullable=False)
    program_id = Column(Integer, ForeignKey("programs.id"), nullable=False)
    applicant_id = Column(Integer, ForeignKey("applicants.id"), nullable=False)
    
    # Personal Information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    phone = Column(String(20))
    date_of_birth = Column(DateTime)
    nationality = Column(String(100))
    
    # Academic Information
    gpa = Column(Float)
    test_scores = Column(JSON)  # SAT, GRE, TOEFL, etc.
    education_history = Column(JSON)  # List of previous education
    work_experience = Column(JSON)  # List of work experience
    
    # Documents
    documents = Column(JSON)  # References to uploaded documents
    personal_statement = Column(Text)
    recommendation_letters = Column(JSON)
    
    # Application Status
    status = Column(String(50), default="submitted")  # submitted, under_review, accepted, rejected, waitlisted
    submission_date = Column(DateTime, default=datetime.utcnow)
    review_date = Column(DateTime, nullable=True)
    decision_date = Column(DateTime, nullable=True)
    
    # AI Analysis
    ai_score = Column(Float, nullable=True)  # Overall AI score (0-100)
    ai_confidence = Column(Float, nullable=True)  # AI confidence level
    ai_analysis = Column(JSON)  # Detailed AI analysis
    ai_recommendation = Column(String(50))  # AI recommendation
    
    # Review
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewer_notes = Column(Text)
    final_decision = Column(String(50))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    program = relationship("Program", back_populates="applications")
    applicant = relationship("Applicant", back_populates="applications")
    reviewer = relationship("User", foreign_keys=[reviewer_id])
    status_history = relationship("ApplicationStatus", back_populates="application")
    
    class Config:
        """Pydantic config."""
        from_attributes = True

class ApplicationStatus(Base):
    """Application status history."""
    
    __tablename__ = "application_status"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"))
    status = Column(String(50))
    notes = Column(Text)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    application = relationship("Application", back_populates="status_history")

# Pydantic Models
class ApplicationBase(BaseModel):
    """Base application schema."""
    
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    nationality: Optional[str] = None
    gpa: Optional[float] = None
    test_scores: Optional[Dict[str, Any]] = None
    education_history: Optional[list] = None
    work_experience: Optional[list] = None
    personal_statement: Optional[str] = None
    program_id: int
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('gpa')
    def validate_gpa(cls, v):
        """Validate GPA range."""
        if v is not None and (v < 0 or v > 4.0):
            raise ValueError('GPA must be between 0 and 4.0')
        return v

class ApplicationCreate(ApplicationBase):
    """Application creation schema."""
    pass

class ApplicationUpdate(BaseModel):
    """Application update schema."""
    
    status: Optional[str] = None
    reviewer_notes: Optional[str] = None
    final_decision: Optional[str] = None

class ApplicationResponse(ApplicationBase):
    """Application response schema."""
    
    id: int
    application_id: str
    status: str
    submission_date: datetime
    ai_score: Optional[float] = None
    ai_recommendation: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        """Pydantic config."""
        from_attributes = True
