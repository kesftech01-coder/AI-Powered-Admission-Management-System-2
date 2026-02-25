"""Applications API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime

from src.core.database import get_db
from src.models.application import Application, ApplicationCreate, ApplicationUpdate, ApplicationResponse
from src.services.ai_evaluator import ai_evaluator
from src.services.document_processor import document_processor
from src.services.notification import notification_service
from src.core.auth import get_current_user
from src.models.user import User

router = APIRouter()

@router.post("/", response_model=ApplicationResponse)
async def create_application(
    application: ApplicationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create new application."""
    # Generate unique application ID
    app_id = f"APP{datetime.utcnow().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"
    
    # Create application record
    db_application = Application(
        application_id=app_id,
        **application.dict(),
        created_by=current_user.id
    )
    
    db.add(db_application)
    db.commit()
    db.refresh(db_application)
    
    # Trigger AI evaluation in background
    background_tasks.add_task(
        process_application_ai,
        db_application.id,
        db
    )
    
    return db_application

@router.get("/", response_model=List[ApplicationResponse])
async def get_applications(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    program_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all applications with filters."""
    query = db.query(Application)
    
    if status:
        query = query.filter(Application.status == status)
    if program_id:
        query = query.filter(Application.program_id == program_id)
    
    applications = query.offset(skip).limit(limit).all()
    return applications

@router.get("/{application_id}", response_model=ApplicationResponse)
async def get_application(
    application_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get application by ID."""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    return application

@router.put("/{application_id}", response_model=ApplicationResponse)
async def update_application(
    application_id: int,
    application_update: ApplicationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update application."""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    for key, value in application_update.dict(exclude_unset=True).items():
        setattr(application, key, value)
    
    application.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(application)
    
    return application

@router.post("/{application_id}/documents")
async def upload_document(
    application_id: int,
    file: UploadFile = File(...),
    document_type: str = "general",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload document for application."""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Process document
    result = await document_processor.process_document(file, document_type)
    
    # Update application documents
    documents = application.documents or []
    documents.append({
        'filename': file.filename,
        'type': document_type,
        'uploaded_at': datetime.utcnow().isoformat(),
        'analysis': result
    })
    application.documents = documents
    
    db.commit()
    
    return {"message": "Document uploaded successfully", "analysis": result}

@router.post("/{application_id}/evaluate")
async def evaluate_application(
    application_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Trigger AI evaluation for application."""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    background_tasks.add_task(
        process_application_ai,
        application_id,
        db
    )
    
    return {"message": "AI evaluation started"}

@router.post("/{application_id}/decision")
async def make_decision(
    application_id: int,
    decision: str,
    notes: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Make final decision on application."""
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    application.final_decision = decision
    application.reviewer_notes = notes
    application.reviewer_id = current_user.id
    application.decision_date = datetime.utcnow()
    application.status = "decided"
    
    db.commit()
    
    # Send notification
    await notification_service.send_decision_notification(
        application.email,
        application.first_name,
        decision
    )
    
    return {"message": f"Application {decision}"}

async def process_application_ai(application_id: int, db: Session):
    """Background task for AI evaluation."""
    application = db.query(Application).filter(Application.id == application_id).first()
    if application:
        # Run AI evaluation
        result = await ai_evaluator.evaluate_application(application)
        
        # Update application with AI results
        application.ai_score = result.get('overall_score')
        application.ai_confidence = result.get('confidence')
        application.ai_analysis = result.get('analysis')
        application.ai_recommendation = result.get('recommendation')
        
        db.commit()
