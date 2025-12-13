"""
Database models and session management.

This module provides database models for persistent storage.
For simple deployments, SQLite is used. For production, PostgreSQL is recommended.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from neural.api.config import settings

Base = declarative_base()


class APIKey(Base):
    """API key model."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<APIKey(name='{self.name}', key='{self.key[:8]}...', is_active={self.is_active})>"


class JobRecord(Base):
    """Job record model."""
    __tablename__ = "jobs"
    
    id = Column(String(255), primary_key=True, index=True)
    type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    user_id = Column(String(255), nullable=True)
    
    def __repr__(self):
        return f"<JobRecord(id='{self.id}', type='{self.type}', status='{self.status}')>"


class DeploymentRecord(Base):
    """Deployment record model."""
    __tablename__ = "deployments"
    
    id = Column(String(255), primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), nullable=False)
    backend = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    endpoint = Column(String(500), nullable=True)
    config = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    user_id = Column(String(255), nullable=True)
    
    def __repr__(self):
        return f"<DeploymentRecord(id='{self.id}', name='{self.name}', status='{self.status}')>"


engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
